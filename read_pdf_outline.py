from pypdf import PdfReader

def print_item(item, depth=0):
    if isinstance(item, list):
        for sub_item in item:
            print_item(sub_item, depth + 1)
    else:
        print("  " * depth + f"- {item.title}")

def print_pdf_outline():
    pdf_path = "books/Hands-On Large Language Models_ Language Understanding and -- Jay Alammar, Maarten Grootendorst -- 1, 2024 -- O'Reilly Media, Incorporated.pdf"
    
    try:
        reader = PdfReader(pdf_path)
        outline = reader.outline
        if outline:
            print("\n--- Recursive Table of Contents ---")
            print_item(outline)
        else:
            print("No outline found.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print_pdf_outline()
