#Flask
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from utils import *

app = Flask(__name__)

# Define the upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET'])
def home():
    return render_template('generate_questions.html')


MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024
@app.route('/process_and_generate_questions', methods=['POST'])
def process_and_generate_questions_api():
    try:
        input_option = request.form.get('input_option')

        if input_option == 'text':
            text_data = request.form.get('input_text')
            if not text_data:
                raise ValueError("Text data is empty")
            result, error_message = process_and_generate_questions(_path=None, _link=None, text=text_data)

        elif input_option == 'file':
            file = request.files['input_file']
            if not file:
                raise ValueError("File is not provided")
            
            # Check file size before saving
            if file.content_length > MAX_FILE_SIZE_BYTES:
                raise ValueError("File size exceeds the limit of 5 megabytes.")

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(file_path)

            result, error_message = process_and_generate_questions(_path=file_path, _link=None, text=None)

            # Check if the file was saved successfully
            if not os.path.exists(file_path):
                raise FileNotFoundError("Failed to save the uploaded file.")

        elif input_option == 'link':
            link = request.form.get('input_link')
            if not link:
                raise ValueError("Link is not provided")
            result, error_message = process_and_generate_questions(_path=None, _link=link, text=None)

        else:
            return jsonify({'success': False, 'message': 'Invalid input option'})

        if error_message:
            response = {'success': False, 'message': error_message}
        else:
            modified_result = {
                'context': result['N_text_file'],
                'questions': [
                    {
                        'answer': entry[0],
                        'distractors': entry[1],
                        'question': entry[2]
                    }
                    for entry in result['keyword_question_distractors']
                ]
            }
            
            response = {
                'success': True,
                'message': 'Questions generated successfully.',
                'result': modified_result
            }

    except ValueError as ve:
        response = {'success': False, 'message': str(ve)}

    except Exception as e:
        response = {'success': False, 'message': f'An unexpected error occurred: {str(e)}'}

    finally:
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)