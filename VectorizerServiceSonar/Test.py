from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from memory_profiler import profile


@profile
def main():
    t2vec_model = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder",
                                           tokenizer="text_sonar_basic_encoder")
    sentences = ['My name is SONAR.',
    'I can embed the sentences into vectorial space.',
    'The output displays the memory consumed by each line in the code.', 
    'Implementation of finding the memory consumption is very easy using a memory profiler as we directly call the decorator instead of writing a whole new code. ']
    
    print(t2vec_model.predict(sentences, source_lang="eng_Latn"))
 
if __name__ == '__main__':
    main()
    
