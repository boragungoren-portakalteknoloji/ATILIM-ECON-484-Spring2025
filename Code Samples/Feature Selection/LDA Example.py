import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder

# Data is prepared using Week 34 results from TFF Archives https://www.tff.org/default.aspx?pageID=545

# CSV dosyasını yükle (ayırıcı ; ise bunu belirt)
df = pd.read_csv("./Teams and Classes.csv", sep=";")

# Etiketli veriler (2022–2023 sezonları)
train_df = df[df["LABEL"].notnull()].copy()

# Etiketsiz veriler (2024 sezonu)
test_df = df[df["LABEL"].isnull()].copy()

# Özellik sütunları (X)
# features = ["O", "G", "B", "M", "A", "Y", "AV", "P"]
features = ["G", "B", "M", "A", "Y", "AV", "P"]
X_train = train_df[features]
y_train = train_df["LABEL"]

# Etiketleri sayısal forma dönüştür
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

# LDA modelini eğit
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train_encoded)

# 2024 takımları için tahmin
X_test = test_df[features]
y_pred_encoded = lda.predict(X_test)

# Tahminleri orijinal etiketlere dönüştür
y_pred_labels = le.inverse_transform(y_pred_encoded)

# Tahminleri dataframe'e ekle
test_df["PREDICTED_LABEL"] = y_pred_labels

# Sonuçları birleştir (etiketli ve tahmin edilenler)
final_df = pd.concat([train_df, test_df], ignore_index=True)

# Yeni CSV olarak kaydet
final_df.to_csv("LDA_results.csv", index=False)

# Ekranda göster
print(test_df[["TEAM", "P", "PREDICTED_LABEL"]])

# Özellik isimleri
feature_names = features

# LDA bileşenleri (LD1, LD2, ...)
components = lda.scalings_[:, :4]  # İlk 4 LD için katsayılar

# LD1–LD4 için her özelliğin katkısını yazdır
for i in range(4):
    print(f"\n📌 LD{i+1} bileşeni katkıları:")
    for feature, coef in zip(feature_names, components[:, i]):
        print(f"{feature}: {coef:.4f}")


