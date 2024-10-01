import time
import numpy as np
import logging
from inverted_index import InvertedIndex
import os

exec_time_encode = {[]}
exec_time_build = []
logger = logging.getLogger("Benchmark")

DEFAULT_INVERTED_INDEX_STORE_PATH = os.path.join(os.getcwd(), 'vim_inverted_index.bin')
DATASET_BIG_PATH = os.path.join(os.getcwd(), 'voina-i-mir0.txt')

#задаем формат вывода сообщений лога
logging.basicConfig(
    format='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logger.setLevel(logging.INFO)

# читаем документы из файла
big_docs = InvertedIndex.load_documents(DATASET_BIG_PATH)
# строим инвертированный индекс
for i in range(10):
    start = time.perf_counter()
    inverted_index = InvertedIndex.build_inverted_index(big_docs)
    end = time.perf_counter() - start
    exec_time_build.append(end * 1000)

queries = ['Дуб', 'Пьер OR Андрей', 'война AND мир', 'завтра OR будет', 'NOT война']

inverted_index.set_use_skip_list(True)

for i in range(10):
    for query in queries:
        logger.info("Start searching [%s] in encode list", query)
        start = time.perf_counter()
        res_list_encode = inverted_index.process_query_encode(query)
        end = time.perf_counter() - start
        exec_time_encode[query].append(end * 1000) 
        logger.info("Complete search [%s] in encode list", query)



# записываем инвертированный индекс на диск
time_write = []
for i in range(10):
    start = time.perf_counter()
    inverted_index.dump(DEFAULT_INVERTED_INDEX_STORE_PATH)
    end = time.perf_counter() - start
    time_write.append(end * 1000)


# считываем инвертированный индекс из файла
time_load = []
for i in range(10):
    start = time.perf_counter()
    inverted_index_load = InvertedIndex.load(DEFAULT_INVERTED_INDEX_STORE_PATH)
    end = time.perf_counter() - start
    time_load.append(end * 1000)

print(f"Время построения индекса: {np.mean(exec_time_build)} ms")
for q in queries:
    print(f"Время поиска для запроса {q}: {np.mean(exec_time_encode[q])} ms")
print(f"Время сохранения на диск: {np.mean(time_write)} ms")
print(f"Время cчитывания с диска: {np.mean(time_load)} ms")
# print(f'inverted_index_load = {inverted_index_load.inverted_index}')