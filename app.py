"""----------------------------------------------------------------------------
    app.py

    Laurent VOURIOT
    
    last update 17/02/2023
-------------------------------------------------------------------------------"""

from flask import Flask
from flask import render_template
from flask import request
from flask import redirect, url_for

from utils import *

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET': 
        return render_template('index_copy.html', data=df_categories,
                                             error=False)

    elif request.method == 'POST':
        date = 1
        values = {'Date' : [date], 
                  'Espece' : [get_code(request.form['Espece'], 'Espece')],  
                  'BMR_ATCD' :  [get_code(request.form['BMR_ATCD'], 'BMR_ATCD')], 
                  'Prelevement' : [get_code(request.form['Prelevement'], 'Prelevement')],
                  'Direct' : [get_code(request.form['Direct'], 'Direct')],
                  'Culture' : [get_code(request.form['Culture'], 'Culture')], 
                  'Service' : [get_code(request.form['Service'], 'Service')],
                  'Genre' : [get_code(request.form['Genre'], 'Genre')], 
                  'Hopital' : [get_code(request.form['Hopital'], 'Hopital')]}
        
        dataframe = pd.DataFrame(values)
        print(debug_msg(dataframe))
        dataframe.fillna(-1, inplace=True)       
        print(debug_msg(dataframe))

        model = load_model('./saved_models_global/crnn_128_64_64_bce_hopital_input.h5')
        
        prediction = model.predict(dataframe)[0] 
        print(debug_msg(len(prediction)))
        print(debug_msg(antibiotiques))

        pred_dict = {antibiotiques[i] : prediction[i] for i in range(len(antibiotiques)-1)} 
        pred_dict = dict(sorted(pred_dict.items(), key=lambda item: item[1], reverse=True))

        return  render_template('prediction.html', values=request.form,
                                                   prediction=pred_dict,
                                                   graphJSON=render_plot(0, pred_dict))
