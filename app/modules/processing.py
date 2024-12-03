from pyspark.ml.feature import RegexTokenizer, StopWordsRemover
from pyspark.ml.feature import IDFModel, HashingTF
from pyspark.sql import SparkSession

labels = { 1:"World",  2:"Sports",  3:"Business", 4:"Science/Tech",}

spark = SparkSession.builder \
    .master("local[*]") \
    .appName("news-classification") \
    .getOrCreate()

idf_vectorizer = IDFModel.load("idfModel")
hashing_tf = HashingTF.load("hashingTF")
    
def process_text(news):
    df = spark.createDataFrame([(news,)], schema=["news"])  
    tokenizer = RegexTokenizer(inputCol="news", outputCol="words", pattern="\\W")
    df = tokenizer.transform(df)
    stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    df = stopwords_remover.transform(df)
    featurized_data = hashing_tf.transform(df)
    rescaled_data = idf_vectorizer.transform(featurized_data)
    return rescaled_data

