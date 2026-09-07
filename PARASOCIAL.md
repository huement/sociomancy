```text
                         ┌──────────────────┐
                         │ YouTube Comment  │
                         └────────┬─────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │ Normalize / language ID   │
                    └─────────────┬─────────────┘
                                  │
            ┌─────────────────────┼────────────────────┐
            │                     │                    │
            ▼                     ▼                    ▼
       Regex Rules          Semantic Model       Emotion Model
       High Precision       Meaning/Intent        Affect
            │                     │                    │
            └─────────────────────┼────────────────────┘
                                  ▼
                         Weighted Scoring
                                  │
                                  ▼
                     Parasocial Signal Score
```

Changed the approach from a simple regex-based personal-disclosure detector into a more structured parasocial signal analyzer.

What I changed:

Importaant Parasocial Files:

```
src/
│
├─ analysis
│       │   ├── models.py
│       │   └── parasocial.py
│       ├── config
│       │   └── parasocial
│       │       └── en.yaml
├─ sociomancy/metrics_parasocial.py

```

## What it does:

External YAML configuration Moves all regex triggers/weights out of Python so you can modify them without touching the analyzer.
Categorized triggers Separates signals into perceived_intimacy, emotional_reliance, reciprocity_expectation, behavioral_influence, etc.
Weighted scoring A comment gets a 0–1-ish parasocial score instead of simply True/False.
Signal explanations The analyzer records why a comment scored highly, including the category and matched pattern.
Positive + negative signals Generic praise or instructional uses of "you" can reduce false positives.
Linguistic context Tracks first-person, second-person, and creator-specific references as supporting evidence.
Better classifications Comments can be none, weak, moderate, strong, or very_strong.
Better samples Instead of just taking the first 3 matches, the report can surface the strongest and moderate examples.
Category density You can see what type of parasocial behavior is occurring per 1,000 comments.
Compiled regexes Patterns are compiled once rather than repeatedly for every comment.
ONNX abstraction Added a clean place to plug in embedding/emotion models without making the core analyzer dependent on them.

## What's still left to do

The biggest pieces are:

1. Add the ONNX semantic model

This is probably the most valuable next step. Instead of relying entirely on exact phrases, embeddings can recognize things like:

"Your videos feel like talking to an old friend."

and

"I know we've never met, but I feel like I've known you forever."

even though neither necessarily matches a regex.

2. Add ONNX emotion analysis

Use emotion detection as supporting evidence, not as the parasocial classifier itself.

For example:

"Love this camera!"

→ high love/emotion, low parasocial

versus:

"Your videos are the only thing that gets me through some days."

→ emotional signal + creator relationship → high parasocial

3. Create a labeled validation dataset

This is probably the most important step for making the results scientifically defensible.

Something like:

## comment label

"Great video!" 0
"I love this camera." 0
"You've changed my life." 1
"I feel like you're a friend." 1
"You should update your drivers." 0
"Your videos got me through a hard time." 1

Then test the detector against hundreds/thousands of real comments.

4. Tune the weights and thresholds

Right now the weights are reasonable starting points, not validated statistical values.

The validation dataset lets us determine things like:

Does perceived_intimacy really deserve 2.5?
Is 0.65 actually a good "strong" threshold?
Which regexes generate false positives?
Which parasocial comments are we missing?

5. Add multilingual configs

Once the architecture is working:

config/parasocial/
├── en.yaml
├── es.yaml
├── fr.yaml
├── de.yaml
└── ...

The core Python doesn't need to change.

6. Eventually train a lightweight final classifier

The end goal I'd recommend is:

```
    Regex signals +
    Embedding similarity +
    Emotion scores +
    Linguistic features
    ↓
    Feature vector
    ↓
    Lightweight classifier
    ↓
    Parasocial probability
```

That would move you from a hand-built heuristic detector toward an actual data-driven classifier while keeping the system explainable.

So, right now
I'd consider the refactor Phase 1: architecture + heuristics.
The next logical step is Phase 2: ONNX embeddings + emotion model, followed by Phase 3: labeled data + calibration/training.
That progression will let you improve accuracy without repeatedly rewriting the core script.

---

Eventually each comment could produce a feature vector like:

```
                 COMMENT
                    │
       ┌────────────┼────────────┐
       │            │            │
      NLP          Regex        ONNX
       │            │            │
       ▼            ▼            ▼
 first-person   intimacy      embedding
 second-person dependency     similarity
 word count     reciprocity   emotion
                influence
       │            │            │
       └────────────┼────────────┘
                    ▼
             FEATURE VECTOR
                    │
                    ▼
             FINAL CLASSIFIER
                    │
                    ▼
        ┌──────────────────────┐
        │ Parasocial Score     │
        │ 0.87                 │
        │ VERY STRONG          │
        ├──────────────────────┤
        │ intimacy             │
        │ emotional reliance   │
        │ long-term attachment │
        └──────────────────────┘
```

And that final classifier is where I would eventually use the labeled dataset.

You could train something lightweight—logistic regression, XGBoost, a small neural classifier, etc.—on the features produced by your regex + embedding + emotion models. Since you're already working with ONNX, the resulting classifier could also be exported to ONNX.

That would be substantially more robust than trying to make the regex file progressively larger.

One other architectural recommendation

Don't throw away the individual scores when you generate your CSV. I'd keep:

```
parasocial_score
parasocial_classification
parasocial_categories
```

and eventually:

```
semantic_intimacy_score
semantic_dependency_score
semantic_reciprocity_score
emotion_admiration
emotion_love
emotion_gratitude
first_person_ratio
second_person_ratio
creator_reference_count
```

That gives you a rich analytical dataset rather than just a final yes/no.

And importantly, it makes your eventual YouTube analysis much more interesting: you can say not merely "this channel has a lot of parasocial comments", but what kind of parasocial relationship the comments are expressing.
