import spacy
from transformers import pipeline

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Load summarization pipeline from Hugging Face
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def clean_and_split(text):
    # Use spaCy to break text into clean sentences
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]
    return " ".join(sentences)

def summarize_text(text, max_len=130, min_len=30):
    clean_text = clean_and_split(text)
    summary = summarizer(clean_text, max_length=max_len, min_length=min_len, do_sample=False)
    return summary[0]['summary_text']

if __name__ == "__main__":
    # Example long article text
    input_text = """
    Artificial intelligence (AI) is transforming the world in unprecedented ways.
    From healthcare to finance, AI is reshaping industries, enabling automation, 
    and improving decision-making. AI-powered tools can now diagnose diseases, 
    drive cars, recommend content, and even write stories. However, as AI grows 
    more capable, concerns around ethics, bias, and job displacement also arise. 
    Policymakers and technologists are working together to ensure responsible AI 
    development that benefits society at large.
    """

    print("\nOriginal Text:\n", input_text)
    print("\nSummary:\n", summarize_text(input_text))
