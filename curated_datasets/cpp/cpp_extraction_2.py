import os
import json

#Dataset Source: https://github.com/zzarif/AI-Detector/tree/main/data/dataset-source-codes.
#Extensions available in dataset: {'json', 'go', 'dockerfile', 'php', 'cpp', 'cs', 'swift', 'dart', 'css', 'ts', 'kt', 'js', 'jav', 'py', 'sql'}.

def generate_dataset(source_dir="dataset-source-codes"):
    """
    Generates two JSONL datasets (human and AI code) from a source directory.

    Args:
        source_dir (str): The path to the directory containing source_code_XXX folders.
    """
    human_data = []
    ai_data = []

    #CodeBert accepted extensions: Python, Java, JavaScript, PHP, Ruby, Go.
    # List of code file extensions to look for.
    code_extensions = ['cpp']

    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' not found.")
        return
    
    # Loop through each folder in the source directory.
    for folder_name in sorted(os.listdir(source_dir)):
        folder_path = os.path.join(source_dir, folder_name)

        # if it's not a directory and folder path is inconsistent, then continue.
        if not os.path.isdir(folder_path) or not folder_name.startswith("source_code_"):
            continue

        # get the number id of the folder.
        folder_id_str = folder_name.replace("source_code_", "")
        # try:
        #     # folder_id = int(folder_id_str)
        #     pass
        # except ValueError:
        #     print(f"Skipping malformed folder name: {folder_name}")
        #     continue

        # loop through the folder and get the files one by one. 
        for file_name in os.listdir(folder_path):
            base_name, ext = os.path.splitext(file_name)
            ext = ext[1:] # Remove the leading dot from extension.
            
            # Loop through files in the current code folder.
            base_name, ext = os.path.splitext(file_name)
            ext = ext[1:] # Remove the leading dot from extension.

            if ext not in code_extensions:
                continue # Skip files not in extension list.

            file_path = os.path.join(folder_path, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code_content = f.read()
            except Exception as e:
                print(f"Error reading code from {file_path}: {e}")
                continue

            entry = {
                "code": code_content,
            }

            # Human-Written Code
            if base_name == f"source_code_{folder_id_str}":
                # entry["class"] = 0
                entry["writer"] = "Human"
                human_data.append(entry)

            # AI Code (GPT-4-TURBO-00)
            elif f"source_code_{folder_id_str}_gpt-4-turbo_00" in base_name:
                if ext in code_extensions:
                    # entry["class"] = 1
                    entry["writer"] = "AI" 
                    ai_data.append(entry)
            # elif f"source_code_{folder_id_str}_gemini_" in base_name:
            #     # Example for Gemini-generated code
            #     entry["class"] = 1
            #     entry["model"] = "Gemini"
            #     ai_data.append(entry)
            # # Add more `elif` conditions here for other AI models if present
            # # For example: elif f"source_code_{folder_id_str}_claude_" in base_name: ...


    # # Write collected data to JSONL files
    # with open("curated_datasets/cpp/human_cpp_2.jsonl", 'w', encoding='utf-8') as f_human:
    #     for item in human_data:
    #         f_human.write(json.dumps(item) + '\n')

    # with open("curated_datasets/cpp/ai_cpp_2.jsonl", 'w', encoding='utf-8') as f_ai:
    #     for item in ai_data:
    #         f_ai.write(json.dumps(item) + '\n')
    
    with open("curated_datasets/cpp/dataset_2.jsonl", 'w', encoding='utf-8') as f_ai:
        for item in ai_data:
            f_ai.write(json.dumps(item) + '\n')
        for item in human_data:
            f_ai.write(json.dumps(item) + '\n')

    # print("\n--- Dataset Generation Summary ---")
    # print(f"Human code samples written: {len(human_data)}")
    # print(f"AI code samples written: {len(ai_data)}")
    # print("Datasets 'human_cpp_2.jsonl' and 'ai_cpp_2.jsonl' created.")
    # print("----------------------------------\n")

# Run the script by calling the function
if __name__ == "__main__":
    generate_dataset()