#!/usr/bin/env ruby

require 'rumale'
require 'natto'

################################################################
# 予測したいテキストの特徴量を求める
################################################################
def get_predict_featurevector(text)

  # ここの関数は既存のデータを読み込むだけにしたい
  # データ: hashの配列
  # hash_array = [{"Sports"=>1, "are"=>1, "physical"=>1}, {"competitive" => 1}, ...]
  def get_hash_data_from_file(f)
    private def make_data_from_file(file)
      labels = []
      texts = []
      CSV.open(file).each do |csv|
        labels << csv[0].to_i
        texts << csv[1]
      end
      [labels, texts]
    end
    private def make_hash_data(texts)
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
    train_labels, train_texts = make_data_from_file(f)
    train_data = make_hash_data(train_texts)
    #puts train_data
    return train_data
  end

  # ここの関数は既存のデータを読み込むだけにしたい
  # データ: 文字列の配列(学習済みの単語のリスト)
  # words = ["Science", "baseball", ...]
  def get_word_count(text)
    private def keitaiso(txt)
      tmp_words = []
      natto = Natto::MeCab.new
      natto.parse(txt) do |n|
        tmp_words << n.surface
      end
      words = tmp_words.reject{|el| el.empty?}
      return words
    end

    private def get_word_list(file)
      word_list = []
      CSV.open(file).each do |csv|
        word_list << keitaiso(csv[1])
      end
      word_list.flatten.uniq
    end

    # 既に学習させた単語しかデータとしては使えないので、
    # それを判断するために、wordsが必要
    # textに含まれる各単語の個数が、Science 2、various 1 と仮定した場合
    word_count = {}
    word_list = get_word_list("./data/sample.txt")
    keitaiso(text).each do | new_word |
      if word_list.include?(new_word)
        if word_count.has_key?(new_word)
          word_count[new_word] += 1
        else
          word_count[new_word] = 1
        end
      end
    end
    word_count
  end

  hash_array = get_hash_data_from_file("./data/sample.txt")
  word_count = get_word_count(text)

  hash_array.push(word_count)

  encoder = Rumale::FeatureExtraction::HashVectorizer.new
  text_data = encoder.fit_transform(hash_array)
  vectorizer = Rumale::FeatureExtraction::TfidfTransformer.new
  tfidf = vectorizer.fit_transform(text_data)
  feature_vector = tfidf.to_a
  numo_samples = Numo::DFloat.cast([feature_vector[-1]])
  return numo_samples

end

# main
if __FILE__ == $0

  # 学習モデルを読み込み
  model = Marshal.load(File.binread("./model/sample_model.dat"))

  # 予測したいテキスト
  text = "Sports physical activities with competitive or recreational elements"

  start_time = Time.now
  # 正規化(??)
  normalizer = Rumale::Preprocessing::L2Normalizer.new
  new_samples = normalizer.fit_transform(get_predict_featurevector(text))

  # 予想
  puts model.predict(new_samples).to_a
  time_label_prediction = Time.now - start_time
  puts "time for label prediction: #{time_label_prediction}s"

end
