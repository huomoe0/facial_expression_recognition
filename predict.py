import json
import os
import sys

import torch
from PIL import Image
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsPixmapItem, QGraphicsScene, QSizePolicy
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.models import densenet121

import Window


def get_image_info(img_path):
    # Get image information such as format, size, etc.
    img = Image.open(img_path)
    img_format = img.format
    img_size = img.size
    img_mode = img.mode
    return [("Format", img_format), ("Size", img_size), ("Mode", img_mode)]


def convert_to_bmp(img_path):
    # Convert image to BMP format using PIL
    img = Image.open(img_path)
    bmp_path = os.path.splitext(img_path)[0] + ".bmp"
    img.save(bmp_path, "BMP")
    return bmp_path


def get_prop_dict(pre):
    data = {
        'anger': pre[0],
        'disgust': pre[1],
        'fear': pre[2],
        'happiness': pre[3],
        'neutral': pre[4],
        'sadness': pre[5],
        'surprise': pre[6]
    }
    return data


def drawHistogram(pre, save_path):
    data = get_prop_dict(pre)
    classes = list(data.keys())
    probabilities = list(data.values())
    # Plotting the bar chart
    plt.bar(classes, probabilities, color='#a8d8ea')
    plt.xlabel('Emotion Classes')
    plt.ylabel('Probabilities')
    plt.title('Emotion Probabilities')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def drawPieChart(pre, save_path):
    data = get_prop_dict(pre)

    labels = list(data.keys())
    sizes = list(data.values())

    # Plotting the pie chart
    plt.figure()
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Emotion Probabilities (Pie Chart)')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


class MainWindowLogic(QMainWindow, Window.Ui_MainWindow):
    def __init__(self):
        super(MainWindowLogic, self).__init__()
        self.setupUi(self)
        self.select.clicked.connect(self.select_image)
        self.image_scene = QGraphicsScene()
        self.pro_scene = QGraphicsScene()
        self.pie_scene = QGraphicsScene()

        # Initialize QGraphicsView
        self.preview.setScene(self.image_scene)
        self.Histogram.setScene(self.pro_scene)
        self.pie.setScene(self.pie_scene)

        self.data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5924, 0.5461, 0.5246], [0.3553, 0.3625, 0.3698])
        ])

        json_path = './class_indices.json'
        with open(json_path, "r") as json_file:
            self.class_indict = json.load(json_file)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = densenet121(num_classes=7).to(self.device)
        self.model_weight_path = "./densenet121.pth"
        self.model.load_state_dict(torch.load(self.model_weight_path, map_location=self.device))
        self.model.eval()

        # Create and set up the model for the QTableView
        model = QtGui.QStandardItemModel(self.info)
        model.setHorizontalHeaderLabels(["Property", "Value"])
        self.info.setModel(model)  # Set the model for the QTableView
        self.info.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.info.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        size_policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.info.setSizePolicy(size_policy)

        try:
            os.mkdir("out")  # 保存输出文件夹
        except Exception as e:
            print(e)

        self.hist_path = "out/prob.png"
        self.pie_path = "out/pie.png"

    def select_image(self):
        file_dialog = QtWidgets.QFileDialog(None)
        file_path, _ = file_dialog.getOpenFileName(self, 'Select Image', '', 'Images (*.png *.jpg *.bmp *.jpeg)')
        print(file_path)  # Add this line to check the file path
        if file_path:
            self.path.setText("路径: " + file_path)
            bmp_path = convert_to_bmp(file_path)
            self.show_image(bmp_path)
            self.process_image(bmp_path)
            # Remove the generated BMP file
            os.remove(bmp_path)

            image_info = get_image_info(file_path)
            self.update_info_table(image_info)
            self.insertHistogram(self.hist_path)  # Insert the histogram image
            self.insertPieChart(self.pie_path)

    def update_info_table(self, image_info):
        try:
            info = self.info.model()
            if info is None:
                print("Model is None")
                return

            info.setRowCount(len(image_info))
            for row, (_property, value) in enumerate(image_info):
                info.setData(info.index(row, 0), _property)
                info.setData(info.index(row, 1), str(value))
        except Exception as e:
            print("Exception in update_info_table:", e)

    def show_image(self, img_path):
        try:
            # Load and display the image in the preview
            pixmap = QPixmap(img_path)
            print("Pixmap size:", pixmap.size())  # Add this line to check the pixmap size
            if pixmap.isNull():
                print("Error loading image:", img_path)
                return
            # 获取预览图片窗口大小
            preview_width = self.preview.width()
            preview_height = self.preview.height()
            print(preview_height, preview_width)

            pixmap = pixmap.scaled(preview_width, preview_height)  # 图片大小调整
            item = QGraphicsPixmapItem(pixmap)
            item.setPos(0, 0)  # 图片位置设置
            self.image_scene.clear()
            self.image_scene.addItem(item)
        except Exception as e:
            print("Exception:", e)

    def process_image(self, img_path):
        """
        加载图片并调用模型预测
        :param img_path: 图片路径
        :return: None
        """
        img = Image.open(img_path)
        img = self.data_transform(img)
        img = torch.unsqueeze(img, dim=0)  # 增加batch_size维度

        with torch.no_grad():
            o = self.model(img.to(self.device)).cpu()
            output = torch.squeeze(o)  # 去掉batch_size维度
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        result_text = "结果预测：{}   概率: {:.3}".format(self.class_indict[str(predict_cla)], predict[predict_cla].numpy())
        self.result.setText(result_text)

        data = []
        for i in range(len(predict)):
            print("class: {:10}   prob: {:.3}".format(self.class_indict[str(i)], predict[i].numpy()))
            data.append(predict[i].numpy())

        drawHistogram(data, self.hist_path)
        drawPieChart(data, self.pie_path)

    def insertPieChart(self, image_path):
        pixmap = QPixmap(image_path)
        item = QGraphicsPixmapItem(pixmap)
        self.pie_scene.clear()
        self.pie_scene.addItem(item)

    def insertHistogram(self, image_path):
        pixmap = QPixmap(image_path)
        item = QGraphicsPixmapItem(pixmap)
        self.pro_scene.clear()
        self.pro_scene.addItem(item)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindowLogic()
    main_window.show()
    sys.exit(app.exec_())
