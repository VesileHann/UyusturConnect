# Derin Öğrenme Sınıflandırma Modeli

Bu proje, bir derin öğrenme modelinin kullanılarak çeşitli sınıflandırma problemlerinin çözümünü ve farklı makine öğrenimi sınıflandırıcılarının performanslarının karşılaştırılmasını içermektedir.

## İçindekiler

- [Genel Bakış](#genel-bakış)
- [Model Mimarisinin Tanımlanması](#model-mimarisinin-tanımlanması)
- [Kullanılan Fonksiyonlar ve Yöntemler](#kullanılan-fonksiyonlar-ve-yöntemler)
- [Performans Değerlendirmesi](#performans-değerlendirmesi)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Lisans](#lisans)

## Genel Bakış

Bu proje, çeşitli sınıflandırma görevlerini çözmek için derin öğrenme modeli oluşturulmuş ve farklı makine öğrenimi sınıflandırıcılarının performanslarını karşılaştırmıştır. Model, Accuracy (Doğruluk), Balanced Accuracy (Dengelenmiş Doğruluk), ROC AUC ve F1 Score metrikleri kullanılarak değerlendirilmiştir.

## Model Mimarisinin Tanımlanması

Derin öğrenme modeli, Keras'ın Sequential API'si kullanılarak aşağıdaki gibi tanımlanmıştır:

```python
from keras.models import Sequential
from keras.layers import Dense, Dropout

classifier = Sequential()

# Giriş katmanı ve ilk gizli katmanın eklenmesi
classifier.add(Dense(units=16, kernel_initializer='uniform', activation='relu', input_dim=12))

# Overfitting'i önlemek için dropout katmanının eklenmesi
classifier.add(Dropout(rate=0.1))

# İkinci gizli katmanın eklenmesi
classifier.add(Dense(units=16, kernel_initializer='uniform', activation='relu'))

# Overfitting'i önlemek için tekrar dropout katmanının eklenmesi
classifier.add(Dropout(rate=0.1))

# Çıkış katmanının eklenmesi
classifier.add(Dense(units=7, kernel_initializer='uniform', activation='sigmoid'))
```

## Kullanılan Fonksiyonlar ve Yöntemler

- **ReLU Aktivasyon Fonksiyonu:** Giriş ve gizli katmanlarda kullanılarak modelin hızlı ve etkili bir şekilde öğrenmesini sağlar.
- **Sigmoid Aktivasyon Fonksiyonu:** Çıkış katmanında kullanılarak her bir sınıf için 0 ile 1 arasında bir olasılık değeri döndürür.
- **Dropout:** Overfitting'i önlemek için kullanılmıştır.
- **Backpropagation:** Modelin öğrenme sürecinde hata geri yayılımı ile ağırlıklar güncellenir.

## Performans Değerlendirmesi

Model performansı, çeşitli metrikler kullanılarak değerlendirilmiştir:

- **Accuracy (Doğruluk)**
- **Balanced Accuracy (Dengelenmiş Doğruluk)**
- **ROC AUC**
- **F1 Score**

Ayrıca, her modelin eğitimi için geçen süre de değerlendirilmiştir. Detaylı sonuçlar aşağıdaki gibidir:

- **DecisionTreeClassifier:** En yüksek doğruluk (%66) ve en hızlı eğitim süresi (0.02 saniye).
- **LGBMClassifier ve ExtraTreeClassifier:** Yüksek doğruluk (%66) ve iyi dengelenmiş doğruluk (%62).
- **BaggingClassifier ve RandomForestClassifier:** %64 doğruluk ve %61 dengelenmiş doğruluk ile dikkat çekici performans.

## Kurulum

Projeyi yerel makinenize klonlayın:
```bash
git clone https://github.com/kullanıcı_adı/proje_adı.git
```

Gerekli bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
```

## Kullanım

Modeli eğitmek ve değerlendirmek için aşağıdaki adımları izleyin:

1. Verisetini yükleyin ve ön işleme tabi tutun.
2. Derin öğrenme modelini tanımlayın ve eğitin.
3. Performans metriklerini hesaplayın ve diğer modellerle karşılaştırın.

Aşağıda temel bir kullanım örneği verilmiştir:

```python
# Verisetini yükleyin
from sklearn.datasets import load_iris
data = load_iris()
X, y = data.data, data.target

# Modeli eğitin
classifier.fit(X, y, epochs=100, batch_size=10)

# Performans değerlendirmesi
accuracy = classifier.evaluate(X, y)
print(f'Model Accuracy: {accuracy}')
```
