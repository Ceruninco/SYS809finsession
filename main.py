import sklearn
from tensorflow.keras.applications import VGG16
from utils.preprocessing import load_data, extract_descriptors
from utils.correspondances import evaluate_sequences
from tensorboardX import SummaryWriter

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))

prefixes = ['legumes', 'magasin','neige','brain','parc','studio','visages']

for name in prefixes:
    writer = SummaryWriter("snapshots/" + name)
    nameA = name + "A"
    nameB = name + "B"

    data_path = 'SYS809_projet2021_sequences1'
    pathA = data_path + '/' + nameA
    img_shape = [224, 224, 3]
    X_train, y_train = load_data(pathA, img_shape, nameA)
    k = 5

    pathB = data_path + '/' + nameB
    img_shape = [224, 224, 3]
    X_test, y_test = load_data(pathB, img_shape, nameB)

    for k in range(5,30,1):
        train_results_reducted, train_classes = extract_descriptors(X_train, y_train, k, conv_base)
        test_results_reducted, test_classes = extract_descriptors(X_test, y_test, k, conv_base)
        accuracy = evaluate_sequences(train_results_reducted, train_classes, test_results_reducted, test_classes)
        writer.add_scalar("accuracy", accuracy, k)
        print("sequence: ", name, " k: ",k," accuracy: ", accuracy)
    writer.close()