from sentence_transformers import SentenceTransformer
from memory_profiler import profile


@profile
def main():
    model = SentenceTransformer('sentence-transformers/LaBSE')
    sentences = ['My name is SONAR.',
    'I can embed the sentences into vectorial space.',
    'The output displays the memory consumed by each line in the code.', 
    'Implementation of finding the memory consumption is very easy using a memory profiler as we directly call the decorator instead of writing a whole new code. ']
    
    print(model.encode(sentences))
 
if __name__ == '__main__':
    main()
    
