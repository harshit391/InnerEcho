from flask import Flask, render_template, request
import joblib
import numpy as np

loaded_model = joblib.load('model.pkl')
scaler = joblib.load('scaler.save')

app = Flask(__name__)

questions = [
    "I found myself getting upset by quite trivial things.",
    "I was aware of dryness of my mouth.",
    "I couldn't seem to experience any positive feeling at all.",
    "I experienced breathing difficulty (eg, excessively rapid breathing, breathlessness in the absence of physical "
    "exertion).",
    "I just couldn't seem to get going.",
    "I tended to over-react to situations.",
    "I had a feeling of shakiness (eg, legs going to give way).",
    "I found it difficult to relax.",
    "I found myself in situations that made me so anxious I was most relieved when they ended.",
    "I felt that I had nothing to look forward to.",
    "I found myself getting upset rather easily.",
    "I felt that I was using a lot of nervous energy.",
    "I felt sad and depressed.",
    "I found myself getting impatient when I was delayed in any way (eg, elevators, traffic lights, being kept "
    "waiting).",
    "I had a feeling of faintness.",
    "I felt that I had lost interest in just about everything.",
    "I felt I wasn't worth much as a person.",
    "I felt that I was rather touchy.",
    "I perspired noticeably (eg, hands sweaty) in the absence of high temperatures or physical exertion.",
    "I felt scared without any good reason.",
    "I felt that life wasn't worthwhile.",
    "I found it hard to wind down.",
    "I had difficulty in swallowing.",
    "I couldn't seem to get any enjoyment out of the things I did.",
    "I was aware of the action of my heart in the absence of physical exertion (eg, sense of heart rate increase, "
    "heart missing a beat).",
    "I felt down-hearted and blue.",
    "I found that I was very irritable.",
    "I felt I was close to panic.",
    "I found it hard to calm down after something upset me.",
    "I feared that I would be 'thrown' by some trivial but unfamiliar task.",
    "I was unable to become enthusiastic about anything.",
    "I found it difficult to tolerate interruptions to what I was doing.",
    "I was in a state of nervous tension.",
    "I felt I was pretty worthless.",
    "I was intolerant of anything that kept me from getting on with what I was doing.",
    "I felt terrified.",
    "I could see nothing in the future to be hopeful about.",
    "I felt that life was meaningless.",
    "I found myself getting agitated.",
    "I was worried about situations in which I might panic and make a fool of myself.",
    "I experienced trembling (eg, in the hands).",
    "I found it difficult to work up the initiative to do things."
]

tipi_questions = [
    "Extraverted, enthusiastic.",
    "Critical, quarrelsome.",
    "Dependable, self-disciplined.",
    "Anxious, easily upset.",
    "Open to new experiences, complex.",
    "Reserved, quiet.",
    "Sympathetic, warm.",
    "Disorganized, careless.",
    "Calm, emotionally stable.",
    "Conventional, uncreative."
]


@app.route('/', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        answers = [request.form.get(f'q{i}') for i in range(1, 43)]
        tipi_answers = [request.form.get(f'tipi{i}') for i in range(1, 11)]
        education = request.form.get('education')
        urban = request.form.get('urban')
        gender = request.form.get('gender')
        age_group = request.form.get('age_group')
        religion = request.form.get('religion')
        orientation = request.form.get('orientation')
        race = request.form.get('race')
        married = request.form.get('married')

        # Process the collected data as needed
        print("Answers:", answers)
        print("TIPI Answers:", tipi_answers)
        print("Education:", education)
        print("Urban:", urban)
        print("Gender:", gender)
        print("Age Group:", age_group)
        print("Religion:", religion)
        print("Orientation:", orientation)
        print("Race:", race)
        print("Marital Status:", married)

        input_data = np.array(
            answers + tipi_answers + [education, urban, gender, age_group, religion, orientation, race, married],
            dtype=object)

        input_data = scaler.transform([input_data])

        # Make prediction
        prediction = loaded_model.predict(input_data)

        # You can then use the prediction as needed, such as displaying it in the response or storing it in a database
        print("Prediction:", prediction[0])

        return render_template('result.html', result=prediction[0])

    return render_template('form.html', questions=questions, tipi_questions=tipi_questions)


if __name__ == '__main__':
    app.run(debug=True)
