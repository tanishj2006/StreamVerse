# StreamVerse

A Gujarati entertainment assistant. Ask about movies, shows, music or podcasts
in **Gujarati or English**; the system classifies what you want and replies in
**Gujarati**.

Built for a Natural Language Processing course at Shah and Anchor Kutchhi
Engineering College, University of Mumbai.

---

## What this actually is

An **intent classifier** with a chat interface, not a generative chatbot.

```
query → detect script → gu or en model → one of six intents → Gujarati reply
```

Given a query, it predicts one of six entertainment intents and returns a fixed
Gujarati response for that intent. It does not look up real titles, search a
catalogue, or generate text. The classifier is the contribution; the replies are
deterministic templates standing in for the downstream recommendation, search and
help systems a production build would route to.

**The six intents**

| Intent | Example |
|---|---|
| Information Query | How do I turn on subtitles on Netflix? |
| Content Search | Where can I watch the new Shah Rukh Khan film? |
| Movie Recommendation | Suggest a good Gujarati film |
| TV Show Recommendation | Which web series should I watch? |
| Music Recommendation | A calm playlist of Gujarati songs |
| Podcast Recommendation | A good podcast about technology |

---

## Quick start

Requires Python 3.9+.

```bash
git clone <your-repo-url>
cd StreamVerse
chmod +x run.sh
./run.sh
```

Open <http://127.0.0.1:8000>. The first run creates a virtual environment and
installs dependencies; later runs start immediately.

<details>
<summary>Manual setup</summary>

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
cd backend && uvicorn app:app --reload --port 8000
```
</details>

### scikit-learn version

A joblib pickle is not guaranteed to load across scikit-learn versions. If the
server logs an `InconsistentVersionWarning`, predictions may differ from the
notebook's. Pin the training version in `backend/requirements.txt`:

```python
# in the training notebook
import sklearn; print(sklearn.__version__)
```

`GET /api/health` reports the version running locally, so the two can be compared.

---

## Model

TF-IDF features over word unigrams, classified by Multinomial Naive Bayes. Two
models — one per input language — share the same six labels and the same Gujarati
response set. No transformers, no translation at any stage.

**Training data.** 150 Gujarati queries, each double-annotated and adjudicated to
a single intent, plus the parallel 150-query English corpus the Gujarati set was
translated from. Both models use the **same record-level train/test split**, so a
translation pair never straddles training and test.

**Uniform class prior.** `fit_prior=False` is set deliberately. With a 21:1 class
imbalance, the frequency-estimated prior causes the classifier to predict only two
of the six intents. A uniform prior recovers all six at the cost of some
majority-class accuracy — a trade-off worth making when the minority intents are
the point of the product.

**Held-out test set (30 queries):**

| | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Gujarati | 0.867 | 0.685 | 0.849 |
| English | 0.767 | 0.566 | 0.777 |
| Majority-class baseline | 0.553 | — | — |

Macro F1 is the primary metric; accuracy is inflated because one intent is 55% of
the data.

> **These numbers are not precise.** 150 records total, 30 in test, with three
> classes contributing a single test row each. Cross-validation and the held-out
> split rank the two languages in *opposite* orders, which is itself evidence that
> the dataset cannot separate them. Neither language is demonstrably easier.

---

## Interface

- **Live intent ladder.** All six intents stay visible in fixed order, so between
  turns only the bars move. A flat distribution looks flat.
- **Token view.** Shows which words survived preprocessing and reached the model.
- **Unsettled readings.** When the top two intents are within 0.05, the result is
  marked rather than presented as confident.
- **Themes.** Light and dark, following the system until you choose, then
  remembered.

Scores shown are relative, not calibrated confidence. Naive Bayes assumes words
occur independently — false for language — which makes its probabilities
systematically overconfident. The ranking is informative; the number is not.

---

## Layout

```
StreamVerse/
├── backend/
│   ├── app.py              FastAPI — /api/health, /api/predict, serves the frontend
│   ├── preprocessing.py    copied verbatim from the training notebook
│   ├── requirements.txt
│   └── model/
│       └── streamverse_bundle.pkl   both models, vectorizers, stopwords, responses
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── run.sh
└── .gitignore
```

One origin serves the API and the page, so there is no CORS configuration.

**`preprocessing.py` must not be rewritten.** It is a verbatim copy of the
notebook's `preprocess()` and `preprocess_en()`. Training-time and serving-time
preprocessing have to be identical, or accuracy drops silently and the cause is
very hard to find. Stopword lists are read from the model bundle rather than
hardcoded, so they cannot drift.

---

## API

```http
POST /api/predict
{ "text": "મને એક સારી ગુજરાતી ફિલ્મ સૂચવો." }
```

```jsonc
{
  "language": "gu",
  "intent": "Movie Recommendation",
  "response": "…",          // always Gujarati
  "confidence": 0.42,        // relative, not calibrated
  "margin": 0.08,            // gap to the runner-up
  "distribution": [ … ],     // all six, fixed order
  "tokens": ["મને", "સારી", "ફિલ્મ", "સૂચવો"],
  "known_tokens": 3          // how many were in the training vocabulary
}
```

`intent` is `null` when the query has no letters, or when every word was a
stopword. `GET /api/health` reports readiness and vocabulary sizes.

---

## Known limitations

1. **150 records.** Far below what reliable intent classification needs. Results
   shift noticeably with a different random seed.
2. **21:1 class imbalance.** The four Recommendation intents have 4–8 training
   examples each, and they are the functional core of the product.
3. **Expect visible misses.** `"મને એક સારી ગુજરાતી ફિલ્મ સૂચવો"` classifies as
   Information Query. The preset chips in the UI are verified demo paths.
4. **TF-IDF carries no semantics.** Two queries with the same intent and different
   words share no features.
5. **Romanised Gujarati is not handled.** `"mane ek saari movie suggest karo"` is
   all Latin characters, so it routes to the English model, whose vocabulary has
   no Gujarati words. The prediction will be close to arbitrary.
6. **Routing is stated, not validated.** The corpus has almost no code-mixing
   (3 of 150 queries), so there is no held-out data to test the rule against.
7. **Replies are templates.** No entity extraction, no dialogue state, no catalogue.

The most useful next step is more annotated data for the Recommendation intents —
not a larger model.

---

## Credits

Dataset construction and annotation: **Tanish Jain** and **Pearl Jain**.
Inter-annotator agreement: Cohen's κ = 0.562 over 150 records, adjudicated to a
consensus label.

Department of Computer Engineering, Shah and Anchor Kutchhi Engineering College
(affiliated to the University of Mumbai).
