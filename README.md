### Структура проекта:

### Основные notebook'и для подготовки данных и запуска обучения:
- `baseline.ipynb` - обучения базового метода
- `d-and-c.ipynb` - обучения метода [Divide and Conquer](https://arxiv.org/abs/1912.06798)
- `data_preprocess.ipynb` - подготовка данных
- `xbm.ipynb` - обучения метода [XBM](https://arxiv.org/abs/1912.06798)

#### Ветка resnet50:
- `configs` - конфиги для датасетов и обучения
- `dataloader/batch_sampler.py` - BatchSampler для выбора изображений в батч
- `dataset/dataset.py` - класс для работы с датасетом
- `executors/resnet50_trainer.py` - трейнер для базовой модели 
- `losses/triplet_loss.py` - реализация Triplet loss
- `metrics` - функции для подсчета метрик
- `models` - модель ResNet-50 и вспомогательные функции 
- `utils` - класс для логгирования метрик и функции визуализации данных 

#### Ветка divide_and_conquer (исходный [код](https://github.com/CompVis/metric-learning-divide-and-conquer)):

- `divide_and_conquer` - библиотека для работы с методом Divide and Conquer

#### Ветка XBM (исходный [код](https://github.com/msight-tech/research-xbm)):

- `research-xbm` - библиотека для работы с методом XBM

#### Данные доступны по [ссылке](https://www.kaggle.com/datasets/stalkerpor1337/flowersdataset)