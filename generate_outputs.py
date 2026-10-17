"""
generate_outputs.py
-------------------
Generates all implementation-plan output figures (Steps 1-5) as PNG files.
Run:  python generate_outputs.py
"""

import os, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import gridspec
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# Palette & style
BG     = "#0f172a"
PANEL  = "#1e293b"
ACCENT = "#6366f1"
GREEN  = "#10b981"
RED    = "#ef4444"
YELLOW = "#f59e0b"
TEXT   = "#f1f5f9"
MUTED  = "#94a3b8"
BORDER = "#334155"

def dark_fig(figsize=(14,6)):
    return plt.figure(figsize=figsize, facecolor=BG)

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
    if title:  ax.set_title(title,  color=TEXT,  fontsize=13, fontweight="bold", pad=10)
    if xlabel: ax.set_xlabel(xlabel, color=MUTED, fontsize=11)
    if ylabel: ax.set_ylabel(ylabel, color=MUTED, fontsize=11)
    ax.grid(axis="y", color=BORDER, linewidth=0.6, linestyle="--", alpha=0.7)
    return ax

OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)
plt.rcParams.update({"font.family":"DejaVu Sans","text.color":TEXT})

# ------------------------------------------------------------------
# STEP 1 - Dataset preview
# ------------------------------------------------------------------
print("[Step 1] Dataset preview ...")
df = pd.read_csv("phishing_url_dataset.csv")

fig = dark_fig(figsize=(18,9))
fig.suptitle("Step 1 - Dataset Preview  (df.head() and df.shape)", color=TEXT, fontsize=16, fontweight="bold", y=0.97)

n_total = len(df); n_legit = (df["label"]==0).sum(); n_phish = (df["label"]==1).sum()
n_feats = len(df.columns) - 2

ax_stats = fig.add_axes([0.02,0.72,0.22,0.22])
ax_stats.set_facecolor("#1e293b")
for sp in ax_stats.spines.values(): sp.set_edgecolor(BORDER)
ax_stats.set_xticks([]); ax_stats.set_yticks([])
stats = [("Total URLs",f"{n_total:,}"),("Legitimate (0)",f"{n_legit:,}"),("Phishing (1)",f"{n_phish:,}"),("Feature Columns",f"{n_feats}")]
for i,(k,v) in enumerate(stats):
    y0 = 0.82-i*0.22
    ax_stats.text(0.05,y0,k,transform=ax_stats.transAxes,color=MUTED,fontsize=10)
    ax_stats.text(0.95,y0,v,transform=ax_stats.transAxes,color=ACCENT,fontsize=12,fontweight="bold",ha="right")
ax_stats.set_title("df.shape", color=TEXT, fontsize=11, fontweight="bold", pad=6)

ax_bar = fig.add_axes([0.27,0.72,0.18,0.22])
ax_bar.set_facecolor(PANEL)
for sp in ax_bar.spines.values(): sp.set_edgecolor(BORDER)
brs = ax_bar.bar(["Legitimate","Phishing"],[n_legit,n_phish],color=[GREEN,RED],width=0.5,edgecolor=BORDER,linewidth=0.8)
for b,v in zip(brs,[n_legit,n_phish]):
    ax_bar.text(b.get_x()+b.get_width()/2,b.get_height()+5,str(v),ha="center",va="bottom",color=TEXT,fontsize=11,fontweight="bold")
ax_bar.set_ylim(0,max(n_legit,n_phish)*1.2)
ax_bar.set_title("Class Distribution",color=TEXT,fontsize=11,fontweight="bold",pad=6)
ax_bar.tick_params(colors=MUTED); ax_bar.yaxis.set_visible(False)
for sp in ax_bar.spines.values(): sp.set_visible(False)

head5 = df.head(5)
display_cols = ["url_length","hostname_length","path_length","num_dots","num_hyphens","num_digits","has_https","has_ip_address","num_suspicious_keywords","has_brand_in_subdomain_or_path","url_entropy","tld","label"]
head5_disp = head5[display_cols]
ax_tbl = fig.add_axes([0.02,0.03,0.96,0.65])
ax_tbl.set_facecolor(BG); ax_tbl.axis("off")
ax_tbl.set_title("df.head(5)  -  selected columns shown", color=MUTED, fontsize=11, loc="left", pad=8)
col_labels = [c.replace("_"," ") for c in head5_disp.columns]
tbl = ax_tbl.table(cellText=head5_disp.values.tolist(),colLabels=col_labels,loc="center",cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1,2.0)
for (row,col),cell in tbl.get_celld().items():
    if row==0:
        cell.set_facecolor(ACCENT); cell.get_text().set_color(TEXT)
        cell.get_text().set_fontsize(8); cell.get_text().set_fontweight("bold")
    else:
        lv = head5_disp.iloc[row-1]["label"]
        cell.set_facecolor("#2d1b1b" if lv==1 else "#1b2d25")
        cell.get_text().set_color(RED if lv==1 else GREEN)
    cell.set_edgecolor(BORDER)
plt.savefig(f"{OUTDIR}/step1_dataset_preview.png",dpi=150,bbox_inches="tight",facecolor=BG); plt.close()
print("   saved -> outputs/step1_dataset_preview.png")

# ------------------------------------------------------------------
# STEP 2 - Features table
# ------------------------------------------------------------------
print("[Step 2] Feature table ...")
features_info = [
    ("url_length","Total number of characters in the URL"),
    ("hostname_length","Length of the domain name"),
    ("path_length","Length of the path after the domain"),
    ("num_dots","Number of . in the URL"),
    ("num_hyphens","Number of - in the URL"),
    ("num_underscores","Number of _ in the URL"),
    ("num_digits","Number of digits in the URL"),
    ("num_special_chars","Number of non-alphanumeric characters"),
    ("num_subdomains","Number of subdomains"),
    ("url_depth","Number of path levels"),
    ("num_query_params","Number of parameters after ?"),
    ("has_at_symbol","1 if @ is present in the URL"),
    ("has_https","1 if the URL uses HTTPS"),
    ("has_ip_address","1 if an IP address is used as the host"),
    ("has_shortener","1 if a URL shortener is used (bit.ly, tinyurl etc.)"),
    ("num_suspicious_keywords","Count of words like login, verify, secure, update"),
    ("has_brand_in_subdomain_or_path","1 if a brand name appears outside the genuine domain"),
    ("digit_ratio","Digits divided by total URL length"),
    ("url_entropy","Shannon entropy - randomness of characters in URL"),
    ("tld","Top-level domain (.com, .xyz, .tk etc.) [categorical]"),
]
fig = dark_fig(figsize=(16,11))
fig.suptitle("Step 2 - Feature Engineering  -  20 URL Features",color=TEXT,fontsize=16,fontweight="bold",y=0.99)
ax = fig.add_axes([0.02,0.02,0.96,0.94]); ax.axis("off")
tbl = ax.table(
    cellText=[[str(i+1),f,d] for i,(f,d) in enumerate(features_info)],
    colLabels=["#","Feature Name","Description"],
    colWidths=[0.04,0.28,0.68],loc="center",cellLoc="left")
tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1,1.72)
cat_color="#2d1a45"
for (row,col),cell in tbl.get_celld().items():
    cell.set_edgecolor(BORDER)
    if row==0:
        cell.set_facecolor(ACCENT); cell.get_text().set_color(TEXT)
        cell.get_text().set_fontweight("bold"); cell.get_text().set_fontsize(10)
    else:
        fn = features_info[row-1][0]
        if fn=="tld":
            cell.set_facecolor(cat_color); cell.get_text().set_color("#c4b5fd")
        elif fn.startswith("has_"):
            cell.set_facecolor("#1a2d1a"); cell.get_text().set_color(GREEN)
        else:
            cell.set_facecolor(PANEL); cell.get_text().set_color(TEXT)
legend_elems=[
    mpatches.Patch(facecolor=PANEL,edgecolor=BORDER,label="Numerical feature"),
    mpatches.Patch(facecolor="#1a2d1a",edgecolor=BORDER,label="Binary (0/1) feature"),
    mpatches.Patch(facecolor=cat_color,edgecolor=BORDER,label="Categorical feature (one-hot encoded)"),
]
ax.legend(handles=legend_elems,loc="lower right",facecolor=PANEL,edgecolor=BORDER,labelcolor=MUTED,fontsize=9,framealpha=0.9)
plt.savefig(f"{OUTDIR}/step2_features_table.png",dpi=150,bbox_inches="tight",facecolor=BG); plt.close()
print("   saved -> outputs/step2_features_table.png")

# ------------------------------------------------------------------
# STEP 3 - Pre-processing
# ------------------------------------------------------------------
print("[Step 3] Pre-processing ...")
X = df.drop(columns=["url","label"]); y = df["label"]
categorical_cols = ["tld"]; numerical_cols = [c for c in X.columns if c not in categorical_cols]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)
tld_counts = X["tld"].value_counts()

fig = dark_fig(figsize=(18,10))
fig.suptitle("Step 3 - Pre-processing Pipeline",color=TEXT,fontsize=16,fontweight="bold",y=0.98)

ax_code = fig.add_axes([0.02,0.42,0.56,0.52])
ax_code.set_facecolor("#0d1117")
for sp in ax_code.spines.values(): sp.set_edgecolor(BORDER)
ax_code.set_xticks([]); ax_code.set_yticks([])
ax_code.set_title("Pre-processing Code",color=TEXT,fontsize=11,fontweight="bold",pad=8)
code_lines=[
    ("# 1. Drop raw URL column + separate target",MUTED),
    ("X = df.drop(columns=['url', 'label'])",TEXT),
    ("y = df['label']",TEXT),("",TEXT),
    ("# 2. Identify column types",MUTED),
    ("categorical_cols = ['tld']",TEXT),
    ("numerical_cols   = [c for c in X.columns if c not in categorical_cols]",TEXT),("",TEXT),
    ("# 3. One-Hot Encode TLD (handle_unknown='ignore')",MUTED),
    ("preprocessor = ColumnTransformer([",TEXT),
    ("    ('cat', OneHotEncoder(handle_unknown='ignore',",TEXT),
    ("            sparse_output=False), categorical_cols)",TEXT),
    ("], remainder='passthrough')",TEXT),("",TEXT),
    ("# 4. Stratified 80 / 20 split",MUTED),
    ("X_train, X_test, y_train, y_test = train_test_split(",TEXT),
    ("    X, y, test_size=0.20,",TEXT),
    ("    random_state=42, stratify=y",TEXT),(")",TEXT),
]
for i,(line,color) in enumerate(code_lines):
    ax_code.text(0.02,0.97-i*0.048,line,transform=ax_code.transAxes,color=color,fontsize=8.8,fontfamily="monospace",va="top")

ax_split = fig.add_axes([0.62,0.60,0.35,0.32]); ax_split.set_facecolor(PANEL)
for sp in ax_split.spines.values(): sp.set_edgecolor(BORDER)
wedges,texts,autotexts = ax_split.pie([len(X_train),len(X_test)],labels=["Train (80%)","Test (20%)"],
    autopct="%1.0f%%",startangle=90,colors=[ACCENT,YELLOW],explode=[0.04,0.04],
    textprops={"color":TEXT,"fontsize":11},wedgeprops={"edgecolor":BG,"linewidth":2})
for at in autotexts: at.set_fontsize(12); at.set_fontweight("bold"); at.set_color(BG)
ax_split.set_title("Train / Test Split",color=TEXT,fontsize=11,fontweight="bold")

ax_info = fig.add_axes([0.62,0.42,0.35,0.15]); ax_info.set_facecolor(PANEL)
for sp in ax_info.spines.values(): sp.set_edgecolor(BORDER)
ax_info.set_xticks([]); ax_info.set_yticks([])
shape_data=[("X_train shape",f"({len(X_train)}, {X_train.shape[1]})"),("X_test shape",f"({len(X_test)}, {X_test.shape[1]})"),
            ("y_train (phishing)",f"{y_train.sum()}"),("y_test (phishing)",f"{y_test.sum()}")]
for i,(k,v) in enumerate(shape_data):
    y0=0.82-i*0.22
    ax_info.text(0.04,y0,k,transform=ax_info.transAxes,color=MUTED,fontsize=9.5)
    ax_info.text(0.96,y0,v,transform=ax_info.transAxes,color=ACCENT,fontsize=10,fontweight="bold",ha="right")

ax_tld = fig.add_axes([0.02,0.04,0.56,0.34])
style_ax(ax_tld,"Top-12 TLD Distribution in Dataset","Top-Level Domain","Count")
top12 = tld_counts.head(12)
bar_colors = [ACCENT if t in ("com","in","org","net","gov") else YELLOW
              if t not in ("tk","xyz","club","online","live","work","icu","top") else RED for t in top12.index]
brs = ax_tld.bar(top12.index,top12.values,color=bar_colors,edgecolor=BG,linewidth=0.8)
for b in brs:
    ax_tld.text(b.get_x()+b.get_width()/2,b.get_height()+1,str(int(b.get_height())),ha="center",va="bottom",color=MUTED,fontsize=8)
ax_tld.set_xticklabels(top12.index,rotation=35,ha="right",color=MUTED)
ax_tld.legend(handles=[mpatches.Patch(facecolor=ACCENT,label="Trusted TLD"),
    mpatches.Patch(facecolor=YELLOW,label="Neutral TLD"),mpatches.Patch(facecolor=RED,label="High-risk TLD")],
    facecolor=PANEL,edgecolor=BORDER,labelcolor=MUTED,fontsize=8.5)

ax_ohe = fig.add_axes([0.62,0.04,0.35,0.34]); ax_ohe.set_facecolor("#0d1117")
for sp in ax_ohe.spines.values(): sp.set_edgecolor(BORDER)
ax_ohe.set_xticks([]); ax_ohe.set_yticks([])
ax_ohe.set_title("One-Hot Encoding - TLD Example",color=TEXT,fontsize=10,fontweight="bold")
ohe_example=[("TLD","com","org","xyz","tk"),("com","1","0","0","0"),("org","0","1","0","0"),("xyz","0","0","1","0"),("tk","0","0","0","1")]
for ri,row in enumerate(ohe_example):
    for ci,val in enumerate(row):
        clr=ACCENT if ri==0 else (GREEN if val=="1" else MUTED); fw="bold" if ri==0 or val=="1" else "normal"
        ax_ohe.text(0.1+ci*0.2,0.85-ri*0.14,val,transform=ax_ohe.transAxes,color=clr,fontsize=10,fontweight=fw,ha="center",va="center",
                    bbox=dict(facecolor="#1e293b",edgecolor=BORDER,boxstyle="round,pad=0.3",linewidth=0.8))
plt.savefig(f"{OUTDIR}/step3_preprocessing.png",dpi=150,bbox_inches="tight",facecolor=BG); plt.close()
print("   saved -> outputs/step3_preprocessing.png")

# ------------------------------------------------------------------
# STEP 4 - Model Training
# ------------------------------------------------------------------
print("[Step 4] Training Random Forest (n_estimators=200) ...")
preprocessor = ColumnTransformer(
    transformers=[("cat",OneHotEncoder(handle_unknown="ignore",sparse_output=False),categorical_cols)],remainder="passthrough")
rf_clf = RandomForestClassifier(n_estimators=200,max_depth=15,random_state=42,n_jobs=-1)
pipeline = Pipeline(steps=[("preprocessor",preprocessor),("classifier",rf_clf)])
pipeline.fit(X_train,y_train)

fig = dark_fig(figsize=(18,10))
fig.suptitle("Step 4 - Model Training - Random Forest Classifier",color=TEXT,fontsize=16,fontweight="bold",y=0.98)

ax_code = fig.add_axes([0.02,0.42,0.58,0.54]); ax_code.set_facecolor("#0d1117")
for sp in ax_code.spines.values(): sp.set_edgecolor(BORDER)
ax_code.set_xticks([]); ax_code.set_yticks([])
ax_code.set_title("Training Code",color=TEXT,fontsize=11,fontweight="bold",pad=8)
train_code=[
    ("from sklearn.ensemble import RandomForestClassifier",TEXT),
    ("from sklearn.pipeline    import Pipeline",TEXT),
    ("from sklearn.compose     import ColumnTransformer",TEXT),
    ("from sklearn.preprocessing import OneHotEncoder",TEXT),("",TEXT),
    ("# Build preprocessing + model pipeline",MUTED),
    ("preprocessor = ColumnTransformer([",TEXT),
    ("    ('cat', OneHotEncoder(handle_unknown='ignore',",TEXT),
    ("            sparse_output=False), ['tld'])",TEXT),
    ("], remainder='passthrough')",TEXT),("",TEXT),
    ("rf_clf = RandomForestClassifier(",TEXT),
    ("    n_estimators = 200,   # number of trees",MUTED),
    ("    max_depth    = 15,    # max tree depth",MUTED),
    ("    random_state = 42,    # reproducibility",MUTED),
    ("    n_jobs       = -1,    # all CPU cores",MUTED),
    (")",TEXT),("",TEXT),
    ("pipeline = Pipeline(steps=[",TEXT),
    ("    ('preprocessor', preprocessor),",TEXT),
    ("    ('classifier',   rf_clf)",TEXT),
    ("])",TEXT),("",TEXT),
    ("pipeline.fit(X_train, y_train)   # <-- Training!",ACCENT),
    ("# Training completed successfully!",GREEN),
]
for i,(line,color) in enumerate(train_code):
    ax_code.text(0.02,0.97-i*0.038,line,transform=ax_code.transAxes,color=color,fontsize=8.8,fontfamily="monospace",va="top")

ax_hp = fig.add_axes([0.64,0.68,0.34,0.28]); ax_hp.set_facecolor(PANEL)
for sp in ax_hp.spines.values(): sp.set_edgecolor(BORDER)
ax_hp.axis("off"); ax_hp.set_title("Hyperparameters",color=TEXT,fontsize=11,fontweight="bold")
hp_data=[("Algorithm","Random Forest"),("n_estimators","200"),("max_depth","15"),("random_state","42"),("n_jobs","-1 (all cores)"),("TLD encoding","One-Hot (OHE)"),("Train:Test","80 : 20"),("Stratified","Yes")]
tbl_hp = ax_hp.table(cellText=[[k,v] for k,v in hp_data],colLabels=["Parameter","Value"],loc="center",cellLoc="center")
tbl_hp.auto_set_font_size(False); tbl_hp.set_fontsize(9); tbl_hp.scale(1,1.5)
for (row,col),cell in tbl_hp.get_celld().items():
    cell.set_edgecolor(BORDER)
    if row==0: cell.set_facecolor(ACCENT); cell.get_text().set_color(TEXT); cell.get_text().set_fontweight("bold")
    else: cell.set_facecolor("#1e293b" if row%2==0 else "#263348"); cell.get_text().set_color(TEXT if col==0 else YELLOW)

rf_fitted = pipeline.named_steps["classifier"]; pre_fitted = pipeline.named_steps["preprocessor"]
ohe_cols = pre_fitted.named_transformers_["cat"].get_feature_names_out(categorical_cols).tolist()
all_feat_names = ohe_cols + numerical_cols
feat_imp_df = pd.DataFrame({"feature":all_feat_names,"importance":rf_fitted.feature_importances_})
feat_imp_df = feat_imp_df.sort_values("importance",ascending=True).tail(10)
ax_imp = fig.add_axes([0.02,0.04,0.58,0.34])
style_ax(ax_imp,"Top-10 Feature Importances (Random Forest)","Importance Score","")
bh = ax_imp.barh(feat_imp_df["feature"],feat_imp_df["importance"],color=ACCENT,edgecolor=BG,linewidth=0.8)
for b in bh:
    w=b.get_width(); ax_imp.text(w+0.001,b.get_y()+b.get_height()/2,f"{w:.4f}",va="center",color=MUTED,fontsize=8.5)
ax_imp.set_yticklabels(feat_imp_df["feature"],color=TEXT,fontsize=9)
ax_imp.set_xlim(0,feat_imp_df["importance"].max()*1.22)
ax_imp.grid(axis="x",color=BORDER,linewidth=0.6,linestyle="--",alpha=0.7); ax_imp.grid(axis="y",visible=False)

ax_tree = fig.add_axes([0.64,0.04,0.34,0.60]); ax_tree.set_facecolor(PANEL)
for sp in ax_tree.spines.values(): sp.set_edgecolor(BORDER)
ax_tree.set_xlim(0,10); ax_tree.set_ylim(0,10); ax_tree.set_xticks([]); ax_tree.set_yticks([])
ax_tree.set_title("Ensemble: 200 Decision Trees => Vote",color=TEXT,fontsize=10,fontweight="bold")
for ti in range(15):
    cx=1+(ti%5)*1.8; cy=8.5-(ti//5)*3.0; clr=GREEN if ti<10 else RED
    ax_tree.add_patch(mpatches.FancyBboxPatch((cx-0.7,cy-1.0),1.4,1.8,boxstyle="round,pad=0.15",facecolor="#263348",edgecolor=clr,linewidth=1.2))
    ax_tree.text(cx,cy-0.1,f"Tree\n{ti+1}",ha="center",va="center",color=TEXT,fontsize=7.5)
ax_tree.annotate("",xy=(5,0.9),xytext=(5,1.4),arrowprops=dict(arrowstyle="->",color=ACCENT,lw=2))
ax_tree.add_patch(mpatches.FancyBboxPatch((2.5,0.1),5,0.8,boxstyle="round,pad=0.1",facecolor=ACCENT,edgecolor="#818cf8",linewidth=1.5))
ax_tree.text(5,0.5,"Majority Vote => Final Prediction",ha="center",va="center",color=TEXT,fontsize=8.5,fontweight="bold")
plt.savefig(f"{OUTDIR}/step4_model_training.png",dpi=150,bbox_inches="tight",facecolor=BG); plt.close()
print("   saved -> outputs/step4_model_training.png")

# ------------------------------------------------------------------
# STEP 5 - Evaluation
# ------------------------------------------------------------------
print("[Step 5] Evaluation ...")
y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:,1]
acc  = accuracy_score(y_test,y_pred)
prec = precision_score(y_test,y_pred,zero_division=0)
rec  = recall_score(y_test,y_pred,zero_division=0)
f1   = f1_score(y_test,y_pred,zero_division=0)
cm   = confusion_matrix(y_test,y_pred)
cr   = classification_report(y_test,y_pred,target_names=["Legitimate","Phishing"])
print(f"\n   Accuracy : {acc*100:.2f}%")
print(f"   Precision: {prec:.4f}")
print(f"   Recall   : {rec:.4f}")
print(f"   F1-Score : {f1:.4f}")
print(f"\nClassification Report:\n{cr}")

fig = dark_fig(figsize=(18,10))
fig.suptitle("Step 5 - Model Evaluation - Random Forest (n_estimators=200)",color=TEXT,fontsize=16,fontweight="bold",y=0.98)

ax_met = fig.add_axes([0.02,0.55,0.42,0.38])
style_ax(ax_met,"Evaluation Metrics","","Score")
metric_names=["Accuracy","Precision","Recall","F1-Score"]
metric_values=[acc,prec,rec,f1]
metric_colors=[GREEN,ACCENT,YELLOW,"#f97316"]
bm = ax_met.bar(metric_names,metric_values,color=metric_colors,width=0.55,edgecolor=BG,linewidth=0.8)
for b,v in zip(bm,metric_values):
    ax_met.text(b.get_x()+b.get_width()/2,b.get_height()+0.005,f"{v*100:.2f}%",ha="center",va="bottom",color=TEXT,fontsize=12,fontweight="bold")
ax_met.set_ylim(0,1.18); ax_met.set_xticklabels(metric_names,color=TEXT,fontsize=11)
ax_met.axhline(y=1.0,color=BORDER,linewidth=0.8,linestyle="--")

ax_cm = fig.add_axes([0.50,0.50,0.24,0.44]); ax_cm.set_facecolor(PANEL)
for sp in ax_cm.spines.values(): sp.set_edgecolor(BORDER)
ax_cm.set_title("Confusion Matrix",color=TEXT,fontsize=11,fontweight="bold",pad=8)
for i in range(2):
    for j in range(2):
        color=GREEN if i==j else RED; alpha=0.35 if i==j else 0.25
        ax_cm.add_patch(mpatches.FancyBboxPatch((j+0.05,1-i+0.05),0.9,0.85,boxstyle="round,pad=0.05",facecolor=color,alpha=alpha,edgecolor=color))
        ax_cm.text(j+0.5,1-i+0.48,str(cm[i,j]),ha="center",va="center",color=TEXT,fontsize=22,fontweight="bold")
        lbl={(0,0):"TN",(0,1):"FP",(1,0):"FN",(1,1):"TP"}[(i,j)]
        ax_cm.text(j+0.5,1-i+0.17,lbl,ha="center",va="center",color=MUTED,fontsize=10)
ax_cm.set_xlim(0,2); ax_cm.set_ylim(0,2)
ax_cm.set_xticks([0.5,1.5]); ax_cm.set_yticks([0.5,1.5])
ax_cm.set_xticklabels(["Pred: Legit","Pred: Phish"],color=MUTED,fontsize=9)
ax_cm.set_yticklabels(["Phishing\n(Actual)","Legit\n(Actual)"],color=MUTED,fontsize=9)

ax_cr = fig.add_axes([0.76,0.50,0.22,0.44]); ax_cr.set_facecolor(PANEL)
for sp in ax_cr.spines.values(): sp.set_edgecolor(BORDER)
ax_cr.axis("off"); ax_cr.set_title("Classification Report",color=TEXT,fontsize=11,fontweight="bold")
prec0 = precision_score(y_test,y_pred,pos_label=0,zero_division=0)
rec0  = recall_score(y_test,y_pred,pos_label=0,zero_division=0)
f10   = f1_score(y_test,y_pred,pos_label=0,zero_division=0)
cr_data=[
    ["Legitimate",f"{prec0:.2f}",f"{rec0:.2f}",f"{f10:.2f}",str(cm[0].sum())],
    ["Phishing",f"{prec:.2f}",f"{rec:.2f}",f"{f1:.2f}",str(cm[1].sum())],
    ["Accuracy","","",f"{acc:.2f}",str(len(y_test))],
    ["Macro avg",f"{(prec0+prec)/2:.2f}",f"{(rec0+rec)/2:.2f}",f"{(f10+f1)/2:.2f}",str(len(y_test))],
]
tbl_cr = ax_cr.table(cellText=cr_data,colLabels=["Class","Prec","Recall","F1","Supp"],loc="center",cellLoc="center")
tbl_cr.auto_set_font_size(False); tbl_cr.set_fontsize(8.5); tbl_cr.scale(1,1.8)
for (row,col),cell in tbl_cr.get_celld().items():
    cell.set_edgecolor(BORDER)
    if row==0: cell.set_facecolor(ACCENT); cell.get_text().set_color(TEXT); cell.get_text().set_fontweight("bold")
    elif row==1: cell.set_facecolor("#1b2d25"); cell.get_text().set_color(GREEN)
    elif row==2: cell.set_facecolor("#2d1b1b"); cell.get_text().set_color(RED)
    else: cell.set_facecolor("#1e293b"); cell.get_text().set_color(MUTED)

ax_hist = fig.add_axes([0.02,0.06,0.42,0.40])
style_ax(ax_hist,"Predicted Probability Distribution","Phishing Probability Score","Count")
bins=np.linspace(0,1,26)
ax_hist.hist(y_proba[y_test==0],bins=bins,color=GREEN,alpha=0.75,label="Legitimate",edgecolor=BG)
ax_hist.hist(y_proba[y_test==1],bins=bins,color=RED,alpha=0.75,label="Phishing",edgecolor=BG)
ax_hist.axvline(x=0.5,color=YELLOW,linewidth=1.8,linestyle="--",label="Decision threshold (0.5)")
ax_hist.legend(facecolor=PANEL,edgecolor=BORDER,labelcolor=MUTED,fontsize=9)

ax_summ = fig.add_axes([0.50,0.06,0.48,0.40]); ax_summ.set_facecolor(PANEL)
for sp in ax_summ.spines.values(): sp.set_edgecolor(BORDER)
ax_summ.set_xticks([]); ax_summ.set_yticks([])
ax_summ.set_title("Results Summary",color=TEXT,fontsize=11,fontweight="bold")
summary_cards=[("Accuracy",f"{acc*100:.2f}%",GREEN),("Precision",f"{prec:.4f}",ACCENT),("Recall",f"{rec:.4f}",YELLOW),("F1-Score",f"{f1:.4f}","#f97316")]
for i,(name,val,col) in enumerate(summary_cards):
    xo=0.05+(i%2)*0.50; yo=0.55-(i//2)*0.42
    ax_summ.add_patch(mpatches.FancyBboxPatch((xo,yo),0.42,0.32,boxstyle="round,pad=0.02",transform=ax_summ.transAxes,facecolor="#263348",edgecolor=col,linewidth=2,clip_on=False))
    ax_summ.text(xo+0.21,yo+0.22,name,transform=ax_summ.transAxes,ha="center",va="center",color=MUTED,fontsize=10)
    ax_summ.text(xo+0.21,yo+0.09,val,transform=ax_summ.transAxes,ha="center",va="center",color=col,fontsize=18,fontweight="bold")
plt.savefig(f"{OUTDIR}/step5_evaluation.png",dpi=150,bbox_inches="tight",facecolor=BG); plt.close()
print("   saved -> outputs/step5_evaluation.png")

print("\nAll output figures generated successfully!")
print(f"  Model: Accuracy={acc*100:.2f}% | F1={f1:.4f} | Precision={prec:.4f} | Recall={rec:.4f}")
