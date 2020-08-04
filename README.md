# AutoEncoderで学習した潜在ベクトルをクラスタリング

## ファイル構造(未更新)

- vae_make_figure.ipynb(.py) : csvファイルから画像を生成
- vae_preprocess.ipynb(.py) : AutoEncoderの訓練用Datasetを用意
- vae_model.ipynb(.py) : AutoEncoderの各モデルを定義
- vae_train_\[モデル名\].ipynb(.py) : AutoEncoderの訓練
- vae_clustering.ipynb(.py) : 中間出力を潜在ベクトルをクラスタリング
- cnn_dataset.ipynb(.py) : 後続のCNN用の入力

## 後続のCNNに向けたデータの仕様

#### データのPATHと種類について

- データなどのpklファイルはmdata_pklsディレクトリにあります
- imgsのナンバリングは画像のサイズ(imgs64_1.pklなら64 * 64サイズの画像), なお、それぞれのサイズのimgsデータは3つのpklファイルに分割されています、使用する際はこれらを統合してください
- ラベルのナンバリングはクラス数(labels50.pklなら50クラス)
- dates.pklは日付のstringが格納されたlistです
- hidden_vecs64.pklはConvAEの中間層64次元出力ベクトルを並べたndarrayです
- imgs128_1.pkl, imgs128_2.pkl, imgs128_3.pklは、128 * 128の画像データです
- imgs64_1.pkl, imgs64_2.pkl, imgs64_3.pklは、64 * 64の画像データです
- labels50.pklは50クラスのint型ラベルのndarrayです
- labels1000.pklは1000クラスのint型ラベルのndarrayです

#### cnn_datasetクラスについて

cnnに食わせるラベル付きDatasetを作るには、例えば以下のように書くことができます(autoencoder_clustering下で実行する場合)

```
from cnn_dataset import cnn_dataset # ここもpathを適宜変更
data_pkls = "./data_pkls/" # ここは適宜変更
pkl_files = [data_pkls + "imgs128_1.pkl", data_pkls + "imgs128_2.pkl", data_pkls + "imgs128_3.pkl"]
num_clusters = 1000
pkl_labels = pkl_labels = data_pkls + "labels" + str(num_clusters) + ".pkl"
cnn_dataset = cnn_dataset(pkl_files, pkl_labels, classes=num_clusters)
```

## 今後に向けて

- 画像自身の改良(これが最重要な気がする)
- 学習を進めたい(400epoch回したけどまだ足りないような)
- 色々なAutoEncoderを試したい
- KNNでDTW班のように潜在ベクトルで近いものを探してくるとか
- 色々なクラスタリング手法を試したい

## ご指摘等はYangまでお願いしますm(__)m