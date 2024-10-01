import bitstring
import logging
from bitstring import Bits, BitArray, BitStream, pack


logger = logging.getLogger("PForDelta")

def pfordelta_decode( bits_stream: BitStream) -> list:
    """ раскодирование list по алгоритму pfordelta """
    posting_list = []
    # если длина 0, то и читать нечего, возвращаем пустой список
    if bits_stream.length == 0:
        return posting_list
    # Общее кол-во элементов в списке
    # Кол-во сжатых элементов
    # Кол-во бит используемых для каждого сжатого элемента
    count, compress_cnt, bits_count = bits_stream.readlist('int:32, int:32, uint:8')
    # Первый элемент (как есть)
    first_num = bits_stream.read('int:32')
    posting_list.append(first_num)
    # Сжатые эл-ты
    if bits_count > 0:
        last_num = first_num
        for i in range(compress_cnt): #len(posting_list)):
            # первый эл-т пропускаем (записываем его как есть
            if i == 0:
                continue
            # получаем дельту между соседними элементами
            # нужное кол-во бит
            delta = bits_stream.read(bits_count)
            last_num += delta.uint
            # добавляем восстановленное число в список
            posting_list.append(last_num)
    # выравниваем по байту
    bits_stream.bytealign()
    # проверяем, все ли элементы списка обработали
    # если нет, то дабавляем их в конец без сжатия
    if compress_cnt < count:
        for i in range(compress_cnt, count):
            num = bits_stream.read('int:32')
            posting_list.append(num)

    return posting_list


def pfordelta_encode(posting_list):
        """ кодирование list по алгоритму pfordelta """
        # Общее кол-во элементов в списке - int
        # Кол-во сжатых элементов - int
        # Кол-во бит используемых для каждого сжатого элемента (используем байт max - 255)
        # Первый элемент (как есть) - int
        # Сжатые эл-ты (массив бит) - bytearray
        # Исключенные элементы в конце списка (как есть) - int

        # вычисляем индекс списка 80% значений
        index_exeption = round(len(posting_list) * 0.8)
        #print(f'index_exeption = {index_exeption}')
        # подсчитываем кол-во бит которым можно закодировать числа
        inc_step = 0
        bits_count = 0
        compress_cnt = 1
        for i in range(len(posting_list)):
            if i != 0:
                inc_step = posting_list[i] - posting_list[i - 1]
                s = bin(inc_step) #[2:]
                bit_arr = BitArray(bin=s)
                if (bit_arr.len > bits_count) and (i >= index_exeption):
                    # если достигли 80% значений и надо увеличить кол-во байт для хранения
                    # то выходим из цикла, остальные значения будем добавлять как есть
                    break
                compress_cnt = i + 1
                if bit_arr.len > bits_count:
                    bits_count = bit_arr.len

        bit_arr = BitArray()
        if bits_count > 0:
            for i in range(compress_cnt): #len(posting_list)):
                # первый эл-т пропускаем (записываем его как есть
                if i == 0:
                    continue
                # получаем дельту между соседними элементами
                # убираем 2 первых символа - 0b
                # дополняем нулями до размерности bits_count
                s = bin(posting_list[i] - posting_list[i - 1])[2:].zfill(bits_count)
                bit_arr.append('0b' + s)
            # получаем остаток от деления
            remainder = bit_arr.len % 8
            # если остаток не 0, то добиваем до целого байта
            if remainder != 0:
                bit_arr.append('0b' + ''.zfill(8 - remainder))
            #bit_arr.pp(width=80)
        # Общее кол-во элементов в списке
        res_arr = bitstring.pack('int:32', len(posting_list))
        # Кол-во сжатых элементов
        res_arr += bitstring.pack('int:32', compress_cnt)
        # Кол-во бит используемых для каждого сжатого элемента
        res_arr += bitstring.pack('uint:8', bits_count)

        #res_arr.pp('bin8, hex', width=80)
        # Первый элемент (как есть)
        res_arr += bitstring.pack('int:32', posting_list[0])

        # Сжатые эл-ты
        if bits_count > 0:
            res_arr += bit_arr
        #res_arr.pp('bin8, hex', width=80)

        # проверяем, все ли элементы списка обработали
        # если нет, то дабавляем их в конец без сжатия
        if compress_cnt < len(posting_list):
            for i in range(compress_cnt, len(posting_list)):
                res_arr += bitstring.pack('int:32', posting_list[i])

        return res_arr

