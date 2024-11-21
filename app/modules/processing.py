from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.sql import SparkSession

labels = { 0:"World",  1:"Sports",  2:"Business", 
          3:"Science", 4:"Health",  5:"Politics", 
          6:"Entertainment", 7:"Tech"}

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

