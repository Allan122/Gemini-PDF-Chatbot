import langchain
import os
print("\n--- ZOMBIE LOCATOR ---")
print(f"Zombie Version: {langchain.__version__}")
print(f"Zombie Location: {os.path.dirname(langchain.__file__)}")
print("----------------------\n")