from CG_lab import *
from CG_lab.MyImage import *
from CG_lab.test import test

if __name__ == "__main__":
    ### lab №1 ###
    # №1
    # test()  # генерация изображений в папке ./CG_lab/images

    # №2, №3
    # test_lines()

    # file_path, prepare, name = './obj_files/fox.obj', MyImage.fox_prepare, "fox"
    # file_path, prepare, name = './obj_files/deer.obj', MyImage.deer_prepare, "deer"
    file_path, prepare, name = './obj_files/rabbit.obj', MyImage.rabbit_prepare, "rabbit"

    # №4, №5, №6, №7
    # test_obj_model(file_path, prepare=prepare)  # отрисовка вершин 3д модели

    ### lab №2 ###
    # test_fill_triangle()

    ### lab №3 ###
    # test_fill_obj_model(file_path, prepare=prepare, model_name=name)  # граней рандомным цветом
    # triangle_3d(file_path, prepare=prepare, model_name=name)  # отрисовка граней с отсечением нелицевых
    # triangle_3d_with_z(file_path, prepare=prepare, model_name=name)  # отрисовка граней с z-буффером

    ### lab №4 ###
    # поворот модели
    # triangle_3d_with_z(file_path, prepare=prepare, model_name=name, angles=[0, 0, 0])
    # проективное преобразование (реализовано только для кролика)
    # triangle_3d_with_z("./obj_files/rabbit.obj", MyImage.rabbit_projective_prepare, "rabbit")

    ### lab №5 ###
    # затенение Гуро (работает только для кролика и лисы, потому что у оленя в .obj файле нет нормалей вершин)
    triangle_3d_with_z(file_path, prepare=prepare, model_name=name, angles=[0, 0, 0], enable_guro_fading=True)
    # наложение текстуры
    triangle_3d_with_z(file_path, prepare=prepare, model_name=name, angles=[0, 0, 0],
                       enable_textures=True, texture_path='./shrek.jpg')