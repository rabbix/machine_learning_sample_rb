#!/usr/bin/env ruby

require 'rumale'
require 'natto'

################################################################
# 全テキストの特徴量を求める
# テキストデータから[ラベルの配列, 特徴量の配列]を抽出する
################################################################
def extract_labels_and_featurevector(f)
  private def make_data_from_file(file)
    labels = []
    texts = []
    CSV.open(file).each do |csv|
      labels << csv[0].to_i
      texts << csv[1]
    end
    [labels, texts]
  end

  private def make_train_hash_data(texts)
    private def keitaiso(txt)
      tmp_words = []
      natto = Natto::MeCab.new
      natto.parse(txt) do |n|
        tmp_words << n.surface
      end
      words = tmp_words.reject{|el| el.empty?}
      return words
    end
    train_data = []
    texts.each do | text |
      word_count = {}
      keitaiso(text).each do | word |
        if word_count.has_key?(word)
          word_count[word] += 1
        else
          word_count[word] = 1
        end
      end
      train_data << word_count
    end
    return train_data
  end

  private def Numo_labels_and_featurevector(labels, data)
    # label
    encoder = Rumale::FeatureExtraction::HashVectorizer.new
    #print "data: #{data}"
    text_data = encoder.fit_transform(data)
    numo_labels = Numo::Int32.cast(labels)
    # samples
    vectorizer = Rumale::FeatureExtraction::TfidfTransformer.new
    tfidf = vectorizer.fit_transform(text_data)
    feature_vector = tfidf.to_a
    numo_samples = Numo::DFloat.cast(feature_vector)
    return numo_labels, numo_samples
  end

  train_labels, train_texts = make_data_from_file(f)
  train_data = make_train_hash_data(train_texts)
  numo_train_labels, numo_train_fvec = Numo_labels_and_featurevector(train_labels, train_data)
  return numo_train_labels, numo_train_fvec
end


# main
if __FILE__ == $0

  # textデータからラベルと特徴量を抽出する
  # NArrayへの変換(Numo_labels_and_featurevectorの中で変換している)
  train_labels, train_samples = extract_labels_and_featurevector("./data/sample.txt")

  # データ前処理
  # ラベルのエンコーディング、サンプルの正規化など

  # モデルの作成
  #model = Rumale::NearestNeighbors::KNeighborsClassifier.new
  model = Rumale::LinearModel::SVC.new
  model.fit(train_samples, train_labels)

  # 評価(同じデータを評価させてる)
  test_labels  = train_labels
  test_samples = train_samples
  puts model.score(test_samples, test_labels)

  # 学習モデル保存
  File.binwrite("./model/sample_model.dat", Marshal.dump(model))

end
