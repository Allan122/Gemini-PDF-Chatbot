try:
    import langchain
    print(f"✅ LangChain Version: {langchain.__version__}")
    from langchain.chains.question_answering import load_qa_chain
    print("✅ SUCCESS: Chains module found!")
except ImportError as e:
    print(f"❌ ERROR: {e}")
except Exception as e:
    print(f"❌ UNKNOWN ERROR: {e}")