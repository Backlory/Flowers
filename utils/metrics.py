
from sklearn.metrics import classification_report                         
from sklearn.metrics import confusion_matrix


def performence(y_groundtruth, y_pred):
    confu = confusion_matrix(y_groundtruth, y_pred)
    temp = classification_report(y_groundtruth, y_pred, zero_division=1, digits=4, output_dict=True)
    accuracy = temp['accuracy']
    
    return confu, accuracy