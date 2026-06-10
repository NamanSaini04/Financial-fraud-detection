# Resume & Portfolio Writeup

Copy-paste blocks for your CV, LinkedIn, and interviews. Fill in any `[ ]` with your
own links. Never invent metrics — these match the actual project results.

---

## 📌 One-liner (CV header / LinkedIn headline)
> Built a credit-card fraud-detection pipeline (XGBoost, SMOTE) on 284K imbalanced
> transactions — PR-AUC 0.82, with a cost-optimal decision threshold that cuts estimated
> fraud loss ~20% — delivered via an interactive Streamlit dashboard.

---

## 📌 Resume bullets (pick 2–4)

**Financial Fraud Detection Analytics** · Python, scikit-learn, XGBoost, imbalanced-learn, Streamlit
- Built an end-to-end ML pipeline to detect fraud in a severely imbalanced dataset
  (0.17% fraud across 284K transactions), engineering a **leak-safe** preprocessing flow
  (split → scale on train only → resample train only).
- Compared Logistic Regression, Random Forest, XGBoost, and SVM, selecting **XGBoost
  (PR-AUC 0.82)** by optimising **recall and PR-AUC over accuracy**, which is misleading
  at 0.17% prevalence.
- Tuned the decision threshold against an **expected-cost model** (missed fraud vs false
  alarm), recommending a threshold that catches **84% of fraud** and lowers estimated cost
  **~20%** vs the default.
- Ran a feature-engineering experiment, **measured no improvement, and removed the
  features** — demonstrating disciplined, evidence-based model selection.
- Shipped an interactive **Streamlit dashboard** with a live threshold simulator and a
  transaction risk scorer, plus stakeholder and technical reports (Markdown + LaTeX PDF).

---

## 📌 Project description (portfolio site / longer form)
> A complete fraud-analytics workflow on the ULB credit-card dataset, built to show
> judgment rather than just model-fitting. Because the data is 99.83% legitimate, the
> project deliberately ignores accuracy and is measured on recall, precision, PR-AUC and
> dollar cost. After leak-safe cleaning and EDA (which surfaced an ~8× fraud-rate spike at
> 2–4 AM and a card-testing pattern in small amounts), I compared four classifiers and
> selected XGBoost (PR-AUC 0.82). I then converted the model's probabilities into a
> business decision by tuning the threshold against an explicit cost model, cutting
> estimated fraud cost ~20% versus the naive default. The project is packaged with an
> interactive Streamlit dashboard, a stakeholder summary, and a full LaTeX report — and is
> explicit about its limitations (PCA-anonymised features, concept drift, placeholder cost
> assumptions).

---

## 📌 Interview talking points (the stories that land)
1. **"Why not accuracy?"** — At 0.17% fraud, predicting *never fraud* scores 99.8%
   accuracy and catches nothing. I led with recall and PR-AUC instead.
2. **The ROC-AUC trap** — My logistic baseline had ROC-AUC 0.97 but 5.6% precision. Same
   model, PR-AUC exposed it as unusable. Great illustration of metric choice under imbalance.
3. **The cost-based threshold** — The "best" threshold isn't 0.5; it depends on what a
   missed fraud costs vs a false alarm. I parameterised those costs and minimised total
   expected cost rather than hard-coding a number.
4. **The negative result** — I engineered three features, tested them, found they didn't
   help (slightly hurt XGBoost), and removed them. Knowing when *not* to add complexity.
5. **Leakage discipline** — Split first, fit the scaler on train only, resample train only.
   The most common way fraud projects fake great results is resampling the test set.

---

## 📌 Links to fill in
- GitHub repo: [ ]
- Live dashboard / demo GIF: [ ]
- Full report (PDF): `reports/analysis_report.pdf`
