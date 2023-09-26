
import tensorflow_transform as tft
import tensorflow as tf

def preprocessing_fn(inputs):

    HbA1c_level=tft.scale_to_0_1(inputs["HbA1c_level"])
    age=tft.scale_to_0_1(inputs['age'])
    blood_glucose_level=tft.scale_to_0_1(inputs['blood_glucose_level'])
    bmi=tft.scale_to_0_1(inputs['bmi'])
    
   
    heart_disease=inputs['heart_disease']
    hypertension=inputs['hypertension'] 
    diabetes=inputs['diabetes']


    integerized_gender = tft.compute_and_apply_vocabulary( inputs['gender'],num_oov_buckets=1,
      vocab_filename='gender_vocab')

    one_hot_encoded_gender = tf.one_hot(
      integerized_gender,
      depth=tf.cast(tft.experimental.get_vocabulary_size_by_name('gender_vocab') + 1,
                    tf.int32),
      on_value=1.0,
      off_value=0.0)
    
    integerized_smoking = tft.compute_and_apply_vocabulary( inputs['smoking_history'],num_oov_buckets=1,
      vocab_filename='smoking_vocab')

    one_hot_encoded_smoking = tf.one_hot(
      integerized_smoking,
      depth=tf.cast(tft.experimental.get_vocabulary_size_by_name('smoking_vocab') + 1,
                    tf.int32),
      on_value=1.0,
      off_value=0.0)
    

    return{"HbA1c_level":HbA1c_level,"Age":age,"Blood_Glucose_Level":blood_glucose_level,"BMI":bmi,"Heart_Disease":heart_disease,
           "Hypertension":hypertension,"Gender":one_hot_encoded_gender,"Smoking_status":one_hot_encoded_smoking,"Diabetes":diabetes}
    


    