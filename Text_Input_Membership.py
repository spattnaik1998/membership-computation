from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import lower, split, col, explode
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.sql.types import StringType
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.linalg import SparseVector
import numpy as np
import sys

def is_float(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("Usage: top_taxis.py <input_file> <output_file>")
    sys.exit(1)

  input_file1 = sys.argv[1]
  input_file2 = sys.argv[2]

  spark = SparkSession.builder.appName("Assignment4_Ques3").getOrCreate()

  wikipedia_pages = spark.read.text(input_file1)

  words_df = wikipedia_pages.select(explode(split(lower(wikipedia_pages.value), "\s+")).alias("word"))

  words_df = words_df.withColumn("word", F.regexp_replace("word", "[^a-zA-Z]", ""))
  words_df = words_df.filter(words_df.word != "")

  word_counts = words_df.groupBy("word").count()

  sorted_word_counts = word_counts.orderBy(F.desc("count"))

  top_words = sorted_word_counts.limit(20000)

  top_words_array = top_words.select("word").rdd.flatMap(lambda x: x).collect()

  print(top_words_array)

  wiki_df = spark.read.option("header", "false").csv(input_file2)

  wiki_df.head()

  wiki_df = wiki_df.withColumnRenamed("_c0", "docID").withColumnRenamed("_c1", "text")

  wiki_df.head()

  tokenizer = Tokenizer(inputCol="text", outputCol="words")
  words_df = tokenizer.transform(wiki_df)
  words_df = words_df.withColumn("word", explode(col("words"))).drop("words")
  words_df = words_df.withColumn("word", col("word").cast("string"))
  words_df = words_df.withColumn("word", col("word").cast("string"))
  words_df = words_df.withColumn("word", F.regexp_replace("word", "[^a-zA-Z]", ""))
  words_df = words_df.filter(words_df.word != "")

  words_df = words_df.groupBy("docID").agg(F.collect_list("word").alias("words"))

  hashing_tf = HashingTF(inputCol="words", outputCol="raw_features", numFeatures=20000)
  idf = IDF(inputCol="raw_features", outputCol="tf_idf_features")
  pipeline = Pipeline(stages=[hashing_tf, idf])
  model = pipeline.fit(words_df)
  tf_idf_matrix = model.transform(words_df)

  def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1.toArray(), vec2.toArray())
    norm1 = np.linalg.norm(vec1.toArray())
    norm2 = np.linalg.norm(vec2.toArray())
    similarity = dot_product / (norm1 * norm2)
    return similarity

  def getPrediction(textInput, k, model, wiki_df):
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    words_df = tokenizer.transform(spark.createDataFrame([(textInput,)]).toDF("text"))
    hashing_tf = model.stages[0].transform(words_df)
    tf_idf_features = model.stages[1].transform(hashing_tf)

    input_vector = tf_idf_features.select("tf_idf_features").rdd.flatMap(lambda x: x).first()

    wiki_with_similarity = tf_idf_matrix.rdd.map(lambda row: (row.docID, row.tf_idf_features, cosine_similarity(input_vector, row.tf_idf_features)))
    top_k_documents = wiki_with_similarity.takeOrdered(k, key=lambda x: -x[2])
    top_categories = [row.words for row in tf_idf_matrix.filter(tf_idf_matrix.docID.isin([doc[0] for doc in top_k_documents])).select("words").collect()]

    return top_k_documents, top_categories

  words_df.head()

  textInput = "Apple Inc. is an American multinational technology company headquartered in Cupertino, California. They design, manufacture, and sell consumer electronics, computer software, and online services."
  k = 20
  top_k_documents, top_categories = getPrediction(textInput, k, model, tf_idf_matrix)

  print("Top", k, "closest documents:")

  for docID, _, similarity in top_k_documents:
    print("Document ID:", docID, "- Cosine Similarity:", similarity)

  spark.stop()