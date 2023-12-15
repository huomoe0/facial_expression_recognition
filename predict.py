import os
import sys

import torch
from PIL import Image
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsPixmapItem, QGraphicsScene, QSizePolicy
from matplotlib import pyplot as plt
from torchvision.models import densenet121

import Window
import utils


def get_image_info(img_path):
    """
    获取图片信息
    :param img_path: 图片路径
    """
    img = Image.open(img_path)
    return [("Format", img.format), ("Size", img.size), ("Mode", img.mode)]


def convert_to_bmp(img_path, save_path):
    """
    把图片另存为为BMP图像
    :param img_path: 图片路径
    :param save_path: 保存路径
    """
    img = Image.open(img_path)
    img.save(save_path, "BMP")


def drawHistogram(pre, save_path):
    """
    生成直方图图片
    :param pre: 概率列表
    :param save_path: 保存路径
    """
    plt.bar(utils.class_list, pre, color='#a8d8ea')
    plt.xlabel('Emotion Classes')
    plt.ylabel('Probabilities')
    plt.title('Emotion Probabilities')
    plt.xticks(rotation=45, ha='right')  # 旋转x轴标签以便观察
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def drawPieChart(pre, save_path):
    """
    生成饼状图图片
    :param pre: 概率列表
    :param save_path: 保存路径
    """
    plt.figure()
    plt.pie(pre, labels=utils.class_list, autopct='%1.1f%%', startangle=140)
    plt.title('Emotion Probabilities (Pie Chart)')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


class MainWindowLogic(QMainWindow, Window.Ui_MainWindow):
    def __init__(self):
        super(MainWindowLogic, self).__init__()
        self.setupUi(self)
        # 设置QGraphicsView
        self.image_scene = QGraphicsScene()
        self.pro_scene = QGraphicsScene()
        self.pie_scene = QGraphicsScene()
        self.preview.setScene(self.image_scene)
        self.Histogram.setScene(self.pro_scene)
        self.pie.setScene(self.pie_scene)
        # 配置模型
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = densenet121(num_classes=7).to(self.device)
        self.model_weight_path = "./densenet121.pth"
        self.model.load_state_dict(torch.load(self.model_weight_path, map_location=self.device))
        self.model.eval()
        # TableView
        model = QtGui.QStandardItemModel(self.info)
        model.setHorizontalHeaderLabels(["Property", "Value"])
        self.info.setModel(model)  # Set the model for the QTableView
        self.info.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.info.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        size_policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.info.setSizePolicy(size_policy)
        # 部分参数
        try:
            os.mkdir("out")  # 创建输出文件夹
        except Exception as e:
            print(e)
        self.hist_path = "out/prob.png"
        self.pie_path = "out/pie.png"
        self.bmp_path = "out/bmp_pic.bmp"
        # 槽函数
        self.select.clicked.connect(self.select_image)

    def select_image(self):
        """
        选择要预测的图片, 并展示
        """
        file_dialog = QtWidgets.QFileDialog(None)
        file_path, _ = file_dialog.getOpenFileName(self, 'Select Image', '', 'Images (*.png *.jpg *.bmp *.jpeg)')
        print(file_path)  # Add this line to check the file path
        if file_path:
            self.path.setText("路径: " + file_path)
            convert_to_bmp(file_path, self.bmp_path)
            self.show_image(self.bmp_path)
            self.process_image(self.bmp_path)

            image_info = get_image_info(file_path)
            self.update_info_table(image_info)
            self.insertHistogram()  # Insert the histogram image
            self.insertPieChart()

    def update_info_table(self, image_info):
        """
        把信息放入表格
        """
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
        """
        显示图片
        :param img_path: 图片路径
        """
        try:
            pixmap = QPixmap(img_path)
            print("Pixmap size:", pixmap.size())  # Add this line to check the pixmap size
            # 获取预览图片窗口大小
            preview_width, preview_height = self.preview.width(), self.preview.height()
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
        img = utils.data_transform["predict"](img)
        img = torch.unsqueeze(img, dim=0)  # 增加batch_size维度

        with torch.no_grad():
            o = self.model(img.to(self.device)).cpu()
            output = torch.squeeze(o)  # 去掉batch_size维度
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        result_text = "结果预测：{}   概率: {:.3}".format(utils.class_list[predict_cla], predict[predict_cla].numpy())
        self.result.setText(result_text)

        data = []
        for i in range(len(predict)):
            print("class: {:10}   prob: {:.3}".format(utils.class_list[i], predict[i].numpy()))
            data.append(predict[i].numpy())

        drawHistogram(data, self.hist_path)
        drawPieChart(data, self.pie_path)

    def insertPieChart(self):
        pixmap = QPixmap(self.pie_path)
        item = QGraphicsPixmapItem(pixmap)
        self.pie_scene.clear()
        self.pie_scene.addItem(item)

    def insertHistogram(self):
        pixmap = QPixmap(self.hist_path)
        item = QGraphicsPixmapItem(pixmap)
        self.pro_scene.clear()
        self.pro_scene.addItem(item)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindowLogic()
    main_window.show()
    sys.exit(app.exec_())
