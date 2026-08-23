import os
import zipfile

def zip_folder(folder_path, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            # Exclude zip_codebase.py itself and output zip file
            for file in files:
                if file.endswith('.zip') or file == 'zip_codebase.py' or '.venv' in root or '.git' in root:
                    continue
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, rel_path)
    print(f"Zip file successfully created at: {output_path}")

if __name__ == "__main__":
    zip_folder(
        "/Users/chudamaniray/Desktop/research/new transformer scratch",
        "/Users/chudamaniray/Desktop/research/transformer_scratch.zip"
    )
