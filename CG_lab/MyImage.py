import numpy as np
from numpy import cos, sin, pi
from typing import List, Tuple
from PIL import Image as im


class MyImage:
    def __init__(self, height: int, width: int, pixels: List = None):
        self.height = height
        self.width = width
        if pixels is None:
            self.pixels = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            self.pixels = np.array(pixels, dtype=np.uint8)
            if len(self.pixels.shape) != 3 or self.pixels.shape[2] != 3:
                assert ValueError("3 channels must be in each pixel!")

    def set_pixel(self, x: int, y: int, color: Tuple[int, int, int]):
        if self.point_exist(x, y):
            self.pixels[y, x] = color
        else:
            assert f"Point ({x}, {y}) doesn't exist"

    def show(self, window_title: str = 'my image'):
        data = im.fromarray(self.pixels)
        data.show(title=window_title)

    def save(self, filename: str):
        data = im.fromarray(self.pixels)
        data.save(filename)

    def clear(self):
        self.pixels = np.zeros_like(self.pixels)

    def point_exist(self, x: int, y: int):
        if 0 <= x < self.width and 0 <= y < self.height:
            return True
        return False

    # подготовка координат 3д модели лисы
    @staticmethod
    def fox_prepare(x, y, z):
        if isinstance(x, list) and isinstance(y, list) and isinstance(z, list):
            return [int(coord * 7 + 500) for coord in x], \
                [int(-coord * 7 + 700) for coord in y], \
                [int(coord * 7 + 500) for coord in z]
        return np.array([int(x * 7 + 500), int(-y * 7 + 700), -int(z * 7 + 500)])

    # подготовка координат 3д модели оленя
    @staticmethod
    def deer_prepare(x, y, z):
        if isinstance(x, list) and isinstance(y, list) and isinstance(z, list):
            return [int(coord * 0.4 + 500) for coord in x], \
                [int(-coord * 0.4 + 700) for coord in y], \
                [int(coord * 0.4 + 500) for coord in z]
        return np.array([int(x * 0.4 + 500), int(-y * 0.4 + 700), -int(z * 0.4 + 500)])

    # подготовка координат 3д модели кролика
    @staticmethod
    def rabbit_prepare(x, y, z):
        if isinstance(x, list) and isinstance(y, list) and isinstance(z, list):
            return [int(coord * 4500 + 500) for coord in x], \
                [int(-coord * 4500 + 700) for coord in y], \
                [int(coord * 4500 + 500) for coord in z]
        return np.array([int(x * 4500 + 500), int(-y * 4500 + 700), -int(z * 4500 + 500)])

    # проективное преобразование координат 3д модели кролика
    @staticmethod
    def rabbit_projective_prepare(x, y, z):
        xyz = np.array([[x, y, z]]).T
        K = np.array([[10000.0, 0.0, 500.0], [0.0, -10000.0, 500.0], [0.0, 0.0, 1.0]])  # матрица
        t = np.array([[0.005, -0.045, 1.5]]).T
        xyz += t
        ret = np.matmul(K, xyz).T
        ret /= ret[0][2]
        ret[0][2] = z * 3000
        return ret[0]

    # получение бицентрических координат для пикселя (x, y)
    def get_bicentric_coordinates(self, x: int, y: int, x0: float, y0: float,
                                  x1: float, y1: float, x2: float, y2: float):
        try:
            lambda0 = ((x1 - x2) * (y - y2) - (y1 - y2) * (x - x2)) / ((x1 - x2) * (y0 - y2) - (y1 - y2) * (x0 - x2))
            lambda1 = ((x2 - x0) * (y - y0) - (y2 - y0) * (x - x0)) / ((x2 - x0) * (y1 - y0) - (y2 - y0) * (x1 - x0))
            lambda2 = ((x0 - x1) * (y - y1) - (y0 - y1) * (x - x1)) / ((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1))
            return lambda0, lambda1, lambda2
        except ZeroDivisionError as zero_error:
            return 0, 0, 0

    # получение ограничивающего прямоугольника для треугольника
    def get_triangle_bounding_box(self, x0, y0, x1, y1, x2, y2):
        xmin = min(x0, x1, x2)
        ymin = min(y0, y1, y2)
        xmax = max(x0, x1, x2)
        ymax = max(y0, y1, y2)

        if xmin < 0: xmin = 0
        if ymin < 0: ymin = 0
        if xmax > self.width: xmax = self.width
        if ymax > self.height: ymax = self.height
        return int(xmin), int(ymin), int(xmax), int(ymax)

    # заполнение треугольника (дефолтное)
    def fill_triangle(self, triangle: List, color=None):
        x0, y0, _, x1, y1, _, x2, y2, _ = triangle
        if color is None:
            color = (255, 255, 255)
        xmin, ymin, xmax, ymax = self.get_triangle_bounding_box(x0, y0, x1, y1, x2, y2)
        for x in range(xmin, xmax):
            for y in range(ymin, ymax):
                l1, l2, l3 = self.get_bicentric_coordinates(x, y, x0, y0, x1, y1, x2, y2)
                if l1 > 0 and l2 > 0 and l3 > 0:
                    self.set_pixel(x, y, color)

    # заполнение треугольника с использованием z-буффера
    def fill_triangle_z(self, triangle: List, normals: List = None, textures: List = None,
                        color=None, z_buff: np.ndarray = None, enable_guro_fading=False, enable_textures=False,
                        texture_image=None):
        first_color = np.array(color)
        x0, y0, z0, x1, y1, z1, x2, y2, z2 = triangle
        if color is None:
            color = np.arra([255, 255, 255], dtype=float)

        xmin, ymin, xmax, ymax = self.get_triangle_bounding_box(x0, y0, x1, y1, x2, y2)
        for x in range(xmin, xmax):
            for y in range(ymin, ymax):
                lam0, lam1, lam2 = self.get_bicentric_coordinates(x, y, x0, y0, x1, y1, x2, y2)
                # отрисовываем пиксель, если все бицентрические координаты строго больше 0
                if lam0 > 0 and lam1 > 0 and lam2 > 0:
                    z_cap = lam0 * z0 + lam1 * z1 + lam2 * z2
                    # отрисовываем пиксель, если его глубина меньше, чем значение на соответствующей позиции в z-буффере
                    if z_cap < z_buff[y, x]:
                        color = np.array(first_color) # из-за 148 объект color изменяется, возвращаю его прежнее значение

                        # изменяем цвет треугольника на цвет пикселя из изображения текстуры
                        if enable_textures and texture_image is not None and textures is not None:
                            w, h = texture_image.size
                            u0, t0, u1, t1, u2, t2 = textures
                            xt, yt = int(w * (lam0 * u0 + lam1 * u1 + lam2 * u2)), int(h * (lam0 * t0 + lam1 * t1 + lam2 * t2))
                            color = texture_image.getpixel((xt, yt))
                        # изменяем яркость пикселя
                        if enable_guro_fading and normals is not None:
                            n0, n1, n2 = normals
                            light = [0, 0, 1]
                            I0 = -255 * np.dot(light, n0)
                            I1 = -255 * np.dot(light, n1)
                            I2 = -255 * np.dot(light, n2)
                            brightness = I0 * lam0 + I1 * lam1 + I2 * lam2
                            color -= brightness
                        self.set_pixel(x, y, color)
                        z_buff[y, x] = z_cap

    # получение вектора нормали между двумя вершинами
    @staticmethod
    def get_normal_vector(v1, v2):
        try:
            n = np.cross(v1, v2)
            return n
        except ZeroDivisionError as zero_error:
            return 0, 0, 0

    # проверка, является ли грань лицевой
    @staticmethod
    def is_front_face(triangle: List):
        x0, y0, z0, x1, y1, z1, x2, y2, z2 = triangle
        v1 = np.array([x1 - x0, y1 - y0, z1 - z0])
        v2 = np.array([x1 - x2, y1 - y2, z1 - z2])
        n = MyImage.get_normal_vector(v1, v2)
        l = np.array([0, 0, 1])
        try:
            val = np.dot(n, l) / (np.linalg.norm(n) * np.linalg.norm(l))
        except Exception as ex:
            return 0
        return val

# отрисовка линии между двумя точками (базовый способ)
def line_1(x0: int, y0: int, x1: int, y1: int, img: MyImage, color: Tuple[int, int, int], point_num: int = 100):
    t_arr = np.linspace(0, 1, point_num, endpoint=True)
    for t in t_arr:
        x = int(x0 * (1.0 - t) + x1 * t)
        y = int(y0 * (1.0 - t) + y1 * t)
        img.set_pixel(x, y, color)

# улучшение line_1
def line_2(x0: int, y0: int, x1: int, y1: int, img: MyImage, color: Tuple[int, int, int]):
    for x in range(x0, x1 + 1):
        t = (x - x0) / (x1 - x0)
        y = int(y0 * (1.0 - t) + y1 * t)
        img.set_pixel(x, y, color)

def _correct_x_y(x0: int, y0: int, x1: int, y1: int, steep):
    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1
    if x0 > x1:  # make it left−to−right
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    return x0, y0, x1, y1

# улучшение line_2
def line_3(x0: int, y0: int, x1: int, y1: int, img: MyImage, color: Tuple[int, int, int]):
    steep = abs(x1 - x0) < abs(y1 - y0)
    x0, y0, x1, y1 = _correct_x_y(x0, y0, x1, y1, steep)
    for x in range(x0, x1 + 1):
        t = (x - x0) / (x1 - x0)
        y = int(y0 * (1 - t) + y1 * t)
        if steep:
            img.set_pixel(x, y, color)
        else:
            img.set_pixel(y, x, color)



# улучшение line_3
def line_4(x0: int, y0: int, x1: int, y1: int, img: MyImage, color: Tuple[int, int, int]):
    steep = abs(x1 - x0) < abs(y1 - y0)
    x0, y0, x1, y1 = _correct_x_y(x0, y0, x1, y1, steep)
    dx = x1 - x0

    if dx == 0:
        for y in range(y0, y1 + 1):
            image[x0, y] = color
        return image

    dy = y1 - y0
    derror = abs(dy) / dx
    error = 0
    y = y0
    for x in range(x0, x1 + 1):
        if steep:
            img.set_pixel(x, y, color)
        else:
            img.set_pixel(y, x, color)
        error += derror
        if error > 0.5:
            y += 1 if y1 > y0 else -1
            error -= 1

# пример отрисовки линий всеми способами
def test_lines():
    h, w = 200, 200
    image = MyImage(h, w)
    color = (255, 255, 255)
    x_mid, y_mid = 100, 100
    alpha = 2 * pi / 13

    points_x, points_y = [], []
    for i in range(0, 13):
        points_x.append(int(100 + 95 * cos(alpha * i)))
        points_y.append(int(100 + 95 * sin(alpha * i)))

    for x, y in zip(points_x, points_y):
        line_1(x_mid, y_mid, x, y, image, color, point_num=50)
    image.show('first method')
    image.clear()

    for x, y in zip(points_x, points_y):
        line_2(x_mid, y_mid, x, y, image, color)
    image.show('second method')
    image.clear()

    for x, y in zip(points_x, points_y):
        line_3(x_mid, y_mid, x, y, image, color)
    image.show('third method')
    image.clear()

    for x, y in zip(points_x, points_y):
        line_4(x_mid, y_mid, x, y, image, color)
    image.show('fourth method')
    image.clear()

# функция для получения данных из .obj файла
def read_obj(file_path: str):
    vertices = []
    faces = []
    normals = []
    textures = []

    with open(file_path) as file:
        for line in file.readlines():
            data = line.split(' ')
            if data[0] == 'v':
                x, y, z = float(data[1]), float(data[2]), float(data[3])
                vertices.append([x, y, z])
            elif data[0] == 'f':
                package = []
                p0_id, p1_id, p2_id = data[1].split('/')[0], data[2].split('/')[0], data[3].split('/')[0]
                package.append([p0_id, p1_id, p2_id])
                if len(data[1].split('/')) > 1:
                    t0_id, t1_id, t2_id = data[1].split('/')[1], data[2].split('/')[1], data[3].split('/')[1]
                    package.append([t0_id, t1_id, t2_id])
                if len(data[1].split('/')) == 3:
                    n0_id, n1_id, n2_id = data[1].split('/')[2], data[2].split('/')[2], data[3].split('/')[2]
                    package.append([n0_id, n1_id, n2_id])
                faces.append(package)
            elif data[0] == 'vn':
                n1, n2, n3 = float(data[1]), float(data[2]), float(data[3])
                normals.append([n1, n2, n3])
            elif data[0] == 'vt':
                u, v = float(data[1]), float(data[2])
                textures.append([u, v])
    if len(vertices) == 0:
        return None
    else:
        return vertices, faces, normals, textures


# отрисовка рёбер 3д модели
def test_obj_model(file_path: str, prepare=None, model_name: str = 'Obj model'):
    h, w = 1000, 1000
    image = MyImage(h, w)
    color = (255, 255, 255)

    points, faces, _, _ = read_obj(file_path)
    faces = np.array(faces, dtype=int)
    faces -= 1

    for point in points:
        x, y, z = point[0], point[1], point[2]
        x, y, z = prepare(x, y, z)

        if not image.point_exist(x, y):
            continue
        image.set_pixel(x, y, color)
    image.show(f'{model_name} points')
    image.clear()

    for face in faces:
        point_0_id, point_1_id, point_2_id = face[0][0], face[0][1], face[0][2]

        x0, y0, z0 = points[point_0_id]
        x1, y1, z1 = points[point_1_id]
        x2, y2, z2 = points[point_2_id]

        x0, y0, _ = prepare(x0, y0, z0)
        x1, y1, _ = prepare(x1, y1, z1)
        x2, y2, _ = prepare(x2, y2, z2)

        if not (image.point_exist(x0, y0) or image.point_exist(x1, y1) or image.point_exist(x2, y2)):
            continue
        line_4(x0, y0, x1, y1, image, color)
        line_4(x0, y0, x2, y2, image, color)
        line_4(x1, y1, x2, y2, image, color)

    image.show(f'{model_name} faces')


def test_fill_triangle():
    img = MyImage(12, 12)

    triangle_1 = [2.5, 1.5, 0., 5.7, 10.4, 0., 10.3, 5.5, 0.]
    triangle_2 = [0, 0, 0, 11, 0, 0, 13, 13, 0]

    img.fill_triangle(triangle_1)
    img.show('Triangle #1')
    img.clear()

    img.fill_triangle(triangle_2)
    img.show('Triangle #2')
    img.clear()

# окрашивание граней 3д модели рандомным цветом
def test_fill_obj_model(file_path: str, prepare=None, model_name: str = 'Obj model'):
    h, w = 1000, 1000
    image = MyImage(h, w)

    points, faces, _, _ = read_obj(file_path)
    faces = np.array(faces, dtype=int)
    faces -= 1

    for face in faces:
        point_0_id, point_1_id, point_2_id = face[0][0], face[0][1], face[0][2]

        v0 = np.array(points[point_0_id])
        v1 = np.array(points[point_1_id])
        v2 = np.array(points[point_2_id])

        v0 = prepare(*v0)
        v1 = prepare(*v1)
        v2 = prepare(*v2)

        triangle = [*v0, *v1, *v2]

        color = (np.random.randint(0, 255, dtype=np.uint8),
                 np.random.randint(0, 255, dtype=np.uint8),
                 np.random.randint(0, 255, dtype=np.uint8))

        image.fill_triangle(triangle, color)
    image.show(f'{model_name} faces')

# окрашивание граней 3д модели с отсечением нелицевых граней
def triangle_3d(file_path: str, prepare=MyImage.deer_prepare, model_name: str = 'Obj model'):
    h, w = 1000, 1000
    image = MyImage(h, w)

    points, faces, _, _ = read_obj(file_path)
    faces = np.array(faces, dtype=int)
    faces -= 1

    for face in faces:
        point_0_id, point_1_id, point_2_id = face[0][0], face[0][1], face[0][2]

        v0 = np.array(points[point_0_id])
        v1 = np.array(points[point_1_id])
        v2 = np.array(points[point_2_id])

        v0 = prepare(*v0)
        v1 = prepare(*v1)
        v2 = prepare(*v2)

        triangle = [*v0, *v1, *v2]

        val = MyImage.is_front_face(triangle)
        if val > 0:
            color = np.array([255 * val, 0, 0], dtype=np.uint8)
            image.fill_triangle(triangle, color)
    image.show(f'{model_name} front faces')


def triangle_3d_with_z(file_path: str, prepare=None, model_name: str = 'Obj model',
                       angles=None, enable_guro_fading=False, enable_textures=False, texture_path='./texture.jpg'):
    h, w = 1000, 1000
    image = MyImage(h, w)
    z_buff = np.zeros((h, w), dtype=np.float32)
    z_buff += np.finfo('f').max

    if angles is None:
        angles = [0, 0, 0]
    points, faces, normals, textures = read_obj(file_path)
    faces = np.array(faces, dtype=int)
    faces -= 1
    a, b, g = np.radians(angles)
    R1 = np.array([[1, 0, 0], [0, cos(a), sin(a)], [0, -sin(a), cos(a)]])
    R2 = np.array([[cos(b), 0, sin(b)], [0, 1, 0], [-sin(b), 0, cos(b)]])
    R3 = np.array([[cos(g), sin(g), 0], [-sin(g), cos(g), 0], [0, 0, 1]])

    R = np.matmul(R1, R2)
    R = np.matmul(R, R3)
    texture_img = None
    if enable_textures:
        texture_img = im.open(texture_path)

    for face in faces:
        point_0_id, point_1_id, point_2_id = face[0][0], face[0][1], face[0][2]
        texture_list = None
        norm_list = None
        if enable_textures and len(face) > 1:
            texture_0_id, texture_1_id, texture_2_id = face[1][0], face[1][1], face[1][2]
            u0, t0 = np.array(textures[texture_0_id])
            u1, t1 = np.array(textures[texture_1_id])
            u2, t2 = np.array(textures[texture_2_id])
            texture_list = [u0, t0, u1, t1, u2, t2]
        if enable_guro_fading and len(face) == 3:
            norma_0_id, norma_1_id, norma_2_id = face[2][0], face[2][1], face[2][2]
            n0 = np.array(normals[norma_0_id])
            n1 = np.array(normals[norma_1_id])
            n2 = np.array(normals[norma_2_id])
            norm_list = np.array([n0, n1, n2])

        v0 = np.array(points[point_0_id])
        v1 = np.array(points[point_1_id])
        v2 = np.array(points[point_2_id])

        v0 = prepare(*v0)
        v1 = prepare(*v1)
        v2 = prepare(*v2)

        # поворот модели (получается правда коряво)
        v0 = np.matmul(R, v0.reshape(len(v0), -1))
        v1 = np.matmul(R, v1.reshape(len(v0), -1))
        v2 = np.matmul(R, v2.reshape(len(v0), -1))
        v0 = v0.T[0]
        v1 = v1.T[0]
        v2 = v2.T[0]

        triangle = [*v0, *v1, *v2]
        val = MyImage.is_front_face(triangle)
        try:
            color = np.array([255, 255, 255], dtype=float)
            if not enable_guro_fading:
                color = color * val
            image.fill_triangle_z(triangle, norm_list, texture_list, color, z_buff,
                                  enable_guro_fading=enable_guro_fading,
                                  enable_textures=enable_textures,
                                  texture_image=texture_img)
        except Exception as ex:
            pass
    image.show(f'{model_name} z-buffered faces')
