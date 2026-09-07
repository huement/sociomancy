| **Chart ID**     | **Type**                  | **Data Input**                   | **Narrative Purpose**                                                 |
| ---------------- | ------------------------- | -------------------------------- | --------------------------------------------------------------------- |
| dna_radar        | 5-Axis Spider Radar       | benchmark.json ($Z$-scores)      | Shows channel $Z$-score polygon against $0.0\sigma$ niche mean circle |
| peer_ranking     | Ranked Horizontal Bar     | benchmark.json (metrics_summary) | Ranks all 9 channels; highlights target channel in neon accent        |
| resonance_delta  | Paired Grouped Bar        | scorecard.json (P1 & P3)         | Compares Baseline vs. Top-Liked comments with floating delta tags     |
| stress_diverging | Center-Zero Diverging Bar | benchmark.json (stress_delta)    | Shows whether toxicity drops or rises on viral pushes ($\Delta T$)    |
| peer_scatter     | 4-Quadrant Scatter Grid   | scorecard.json (discovery)       | Semantic Similarity (X) vs. Audience Affinity (Y) competitor plot     |
| emotion_donut    | Ring Donut Chart          | scorecard.json (P1 Emotions)     | Displays top emotion distribution with dominant emotion in center     |
| bot_breakdown    | Stacked Horizontal Bar    | scorecard.json (P5 Flags)        | Proportional segments showing exact copy-paste vs. cluster spam       |

### TODO

Word cloud Chart for each channel : [https://github.com/ecomfe/echarts-wordcloud](https://github.com/ecomfe/echarts-wordcloud)

#### CHARTS

Yearly Stats: [https://youtube-analysis-alh0.onrender.com/analysis](https://youtube-analysis-alh0.onrender.com/analysis)

- Total Videos / Average Video Totals
- Total Views / Average Views
- Total Likes / Average Likes
- Total Comments / Average Comments
