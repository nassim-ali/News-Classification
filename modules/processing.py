from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.sql import SparkSession

labels = {"World": 0, "Sports": 1, "Business": 2, 
          "Science": 3, "Health": 4, "Politics": 5, 
          "Entertainment": 6}

spark = SparkSession.builder \
    .master("local[*]") \
    .appName("news-classification") \
    .getOrCreate()
    
def process_text(news):
    df = spark.createDataFrame([(news,)], schema=["news"])  
    tokenizer = RegexTokenizer(inputCol="news", outputCol="words", pattern="\\W")
    df = tokenizer.transform(df)
    stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    df = stopwords_remover.transform(df)
    hashing_tf = HashingTF(inputCol="filtered", outputCol="raw_features", numFeatures=10000)
    featurized_data = hashing_tf.transform(df)
    idf = IDF(inputCol="raw_features", outputCol="features")
    idf_vectorizer = idf.fit(featurized_data)
    rescaled_data = idf_vectorizer.transform(featurized_data)
    return rescaled_data

