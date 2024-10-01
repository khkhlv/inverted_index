from __future__ import annotations
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import logging
from collections import defaultdict
import math
import os
import string
import struct
import logging
from bitstring import BitStream, pack
from pfodelta import pfordelta_decode, pfordelta_encode

logger = logging.getLogger("InvertedIndex")

class InvertedIndex:

    """Inverted index implementation"""
    def __init__(self, documents: list):
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('punkt_tab')
        """Constructor for class InvertedIndex"""
        # обычный список  key -> list[int]
        self.inverted_index = defaultdict(list)
        # закодированный список key -> list[byte]
        self.posting_list_encoded = defaultdict(list)
        # обычный список skip list
        self.inverted_index_skip_list = defaultdict(list)
        # закодированный список skip list
        self.skip_list_encoded = defaultdict(list)
        self.use_skip_list = True
        # сохраняем количество документов (нужно для операции NOT)
        self.num_docs = len(documents)
        logger.info("Start tokenization documents list...")
        for doc_id, doc in enumerate(documents):
            tokens = self.preprocess_text(doc)
            for token in tokens:
                if doc_id not in self.inverted_index[token]:  # Проверяем, чтобы избежать дубликатов
                    self.inverted_index[token].append(doc_id)
        logger.info("Complete tokenization documents list...")
        # Сортировка doc_id для каждого токена
        for token in self.inverted_index:
            self.inverted_index[token].sort()
            # заполняем остальные списки
            self.fill_list(token)
        logger.info("Complete sorting and encoding list...")

    def fill_list(self, token: str):
        """ Заполнение posting_list_encoded, skip_list_encoded
        inverted_index_skip_list для token """
        # шаг skip list
        step_skip_list = self.get_step_skip_list(len(self.inverted_index[token]))
        # заполняем skip list
        len_item_list = len(self.inverted_index[token])
        for i in range(len_item_list):
            if i % step_skip_list == 0:
                # если попадаем на последний элемент в инв индексе,
                # то не добавляем этот эл-т в skip list
                if i + step_skip_list < len_item_list:
                    num = self.inverted_index[token][i + step_skip_list]
                    self.inverted_index_skip_list[token].append(num)
        # кодируем список
        bin_obj = pfordelta_encode(self.inverted_index[token]).bytes
        # сохраняем кодированныы список
        self.posting_list_encoded[token] = bin_obj
        # кодируем skip list и сохраняем
        if len(self.inverted_index_skip_list[token]) > 0:
            bin_obj = pfordelta_encode(self.inverted_index_skip_list[token]).bytes
            self.skip_list_encoded[token] = bin_obj
        return

    def set_use_skip_list(self, value: bool):
        """ Сеттер для use_skip_list """
        self.use_skip_list = value
        return

    def get_step_skip_list(self, length: int) -> int:
        """ вычисляем шаг skip list"""
        # шаг skip list
        # согласно книге - Введение в инф. поиск (стр 58)
        # если длина инв. индекса равна P, то следует использовать
        # sqrt(P) равномерно размещенных указателей пропусков
        return round(math.sqrt(length))

    def hasSkip(self, token: str, index: int) -> bool:
        """ Функция показывает, можем ли использовать skip list"""
        # можно использовать skip list? если нет, то False
        if self.use_skip_list == False:
            return False
        #logger.debug("hasSkip. use_skip_list = [%s] ", self.use_skip_list)
        #logger.debug("hasSkip. token = [%s] ", token)
        # Token пуст? то False
        if len(token) == 0:
            return False
        #logger.debug("hasSkip. len(self.inverted_index_skip_list[token]) = [%s] ", len(self.inverted_index_skip_list[token]))
        # Список пуст? то False
        if len(self.inverted_index_skip_list[token]) == 0:
            return False
        # Возвращаем True, если index в posting_list соответствует значению в skip list
        # шаг skip list
        step_skip_list = self.get_step_skip_list(len(self.inverted_index[token]))
        #logger.debug("hasSkip. step_skip_list = [%s], index = %s ", step_skip_list, index)
        # если попадаем на последний элемент в инв индексе,
        # то возвращаем false
        if index + step_skip_list >= len(self.inverted_index[token]):
            result = False
        else:
            result = index % step_skip_list == 0
        #logger.debug("hasSkip. result = [%s] ", result)
        return  result

    def getSkip(self, token: str, index: int) -> int:
        """ Функция возвращает значение на которое указывает skip list """
        # Перед вызовом этой функции должна вызываться hasSkip которая возвращает True
        result = -1
        logger.debug("getSkip. token = [%s] ", token)
        # шаг skip list
        step_skip_list = self.get_step_skip_list(len(self.inverted_index[token]))
        logger.debug("getSkip. step_skip_list = [%s] ", step_skip_list)
        if index + step_skip_list < len(self.inverted_index[token]):
            # // целочисленное деление
            index_skip_list = index // step_skip_list
            result = self.inverted_index_skip_list[token][index_skip_list]
        logger.debug("getSkip. result = [%s] ", result)
        return result

    def hasSkip_encode(self, token: str, index: int) -> bool:
        """ Функция показывает, можем ли использовать skip list
        работает с кодированным списком """
        # мржно использовать skip list? если нет, то False
        if self.use_skip_list == False:
            return False
        # Token пуст? то False
        if len(token) == 0:
            return False
        # Список пуст? то False
        if self.pl_len(self.skip_list_encoded[token]) == 0:
            return False
        # Возвращаем True, если index в posting_list соответствует значению в skip list
        # шаг skip list
        step_skip_list = self.get_step_skip_list(self.pl_len(self.posting_list_encoded[token]))
        # если попадаем на последний элемент в инв индексе,
        # то возвращаем false
        if index + step_skip_list >= self.pl_len(self.posting_list_encoded[token]):
            result = False
        else:
            result = index % step_skip_list == 0
        return  result

    def getSkip_encode(self, token: str, index: int) -> int:
        """ Функция возвращает значение на которое указывает skip list
        работает с кодированным списком """
        # Перед вызовом этой функции должна вызываться hasSkip которая возвращает True
        result = -1
        logger.debug("getSkip_encode. token = [%s] ", token)
        # шаг skip list
        step_skip_list = self.get_step_skip_list(self.pl_len(self.posting_list_encoded[token]))
        logger.debug("getSkip_encode. step_skip_list = [%s] ", step_skip_list)
        if index + step_skip_list < self.pl_len(self.posting_list_encoded[token]):
            index_skip_list = index // step_skip_list
            logger.debug("getSkip_encode. index_skip_list = [%s], index = %s", index_skip_list, index)
            #result = self.skip_list_encoded[token][index_skip_list]
            result = self.pl_get(self.skip_list_encoded[token], index_skip_list)
            logger.debug("getSkip_encode. result = [%s] ", result)
        return result

    def pl_get(self, encode_val, index):
        """функция возвращает раскодированное значение (index)
        из постинг листа (encode_val)"""
        #print(f'encode_val = {encode_val}')
        posting_list = pfordelta_decode(BitStream(bytes=encode_val))
        #print(f'posting_list = {posting_list}')
        return posting_list[index]

    def pl_len(self, encode_val):
        """Функция возвращает общее количество элементов
        в закодированном списке encode_val"""
        if encode_val is not None:
            bits_stream = BitStream(encode_val)
            # Общее кол-во элементов в списке
            count, compress_cnt, bits_count = bits_stream.readlist('int:32, int:32, uint:8')
        else:
            count = 0
        return count

    def preprocess_text(self, text):
        """ разбираем список """
        # Tokenization
        tokens = word_tokenize(text.lower())

        # Stop-word removal
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]

        # Removing punctuation
        tokens = [token for token in tokens if token not in string.punctuation]

        return tokens

    @classmethod
    def load(cls, filepath: str) -> InvertedIndex:
        """Loads inverted index from file"""
        with open(filepath, 'rb') as file_io:

            #cls.num_docs = 0
            # read count items in inverted index
            bin_obj = file_io.read(4)
            count = struct.unpack('>i', bin_obj)[0]
            inverted_index_load = defaultdict(list)
            inv_index = InvertedIndex(dict())
            # очищаем кол-во документов перед загрузкой
            inv_index.num_docs = 0
            logger.info("Start loading %s tokens of inverted index to file %s", count, filepath)
            #print(f'count={count}')
            bits_stream = BitStream(file_io)
            for i in range(count):
                bin_obj = file_io.read(2)
                length = struct.unpack('>H', bin_obj)[0]
                #print(f'length={length}')
                # format string for read
                fmt = '>' + str(length) + 's'
                bin_obj = file_io.read(length)
                # unpack string in bytes
                byte_str = struct.unpack(fmt, bin_obj)[0]
                # convert bytes in str
                word = byte_str.decode()
                #print(f'word = {word}')

                # в BitStream устанавливаем указатель
                bits_stream.bytepos = file_io.tell()
                #print(f'bits_stream.read(32).hex = {bits_stream.read(32).hex}')
                #print(f'file_io.tell() = {file_io.tell()}')

                posting_list = pfordelta_decode(bits_stream)
                #print(f'posting_list = [{posting_list}]')
                # добавляем восстановленные числа в список
                inverted_index_load[word] = posting_list
                # подсчитываем общее количество документов
                # так как список отсортирован, то сравниваем с последним элементом
                # 1 добавляем, так как документны нумеруются с 0
                if inv_index.num_docs < posting_list[-1] + 1:
                    inv_index.num_docs = posting_list[-1] + 1
                #bits_stream.bytepos = file_io.tell()
                # перемещаем указатель в файле
                file_io.seek(bits_stream.bytepos)
                #print(f'file_io.tell() = {file_io.tell()}')
        # заполняем созданный класс InvertedIndex считанными значениями
        for key, value in inverted_index_load.items():
            inv_index.inverted_index[key] = value
            # заполняем остальные списки
            inv_index.fill_list(key)
        logger.info("Complete loading %s tokens of inverted index to file %s", count, filepath)
        return inv_index

    def dump(self, filepath: str) -> None:
        """Save inverted index to file"""
        with open(filepath, 'wb') as file_io:
            # write count items in index
            count = len(self.inverted_index)
            logger.info("Start saving %s tokens of inverted index to file %s", count, filepath)
            # get packing data
            bin_obj = struct.pack('>i', count)
            # store packing data
            file_io.write(bin_obj)
            i = 0
            for word, article_list in self.inverted_index.items():
                # store word length
                byte_str = word.encode()
                length = len(byte_str)
                bin_obj = struct.pack('>H', length)
                file_io.write(bin_obj)
                # store word
                bin_obj = struct.pack('>' + str(length) + 's', byte_str)
                file_io.write(bin_obj)
                # записываем список
                bin_obj = pfordelta_encode(article_list).bytes
                file_io.write(bin_obj)
            logger.info("Complete saving %s tokens of inverted index to file %s", count, filepath)
        return
    
    def AND(self, posting1, posting2):
        """ AND оператор для list """
        # [0] список, [1] - token
        key1 = posting1[1]
        key2 = posting2[1]
        posting1 = posting1[0]
        posting2 = posting2[0]
        p1 = 0
        p2 = 0
        result = list()
        if (posting1 is not None) and (posting2 is not None):
            while p1 < len(posting1) and p2 < len(posting2):
                if posting1[p1] == posting2[p2]:
                    result.append(posting1[p1])
                    p1 += 1
                    p2 += 1
                elif posting1[p1] > posting2[p2]:
                    # поиск по skip list
                    if self.hasSkip(key2, p2) and (self.getSkip(key2, p2) < posting1[p1]):
                        while self.hasSkip(key2, p2) and (self.getSkip(key2, p2) < posting1[p1]):
                            p2 += self.get_step_skip_list(len(self.inverted_index[key2]))
                            logger.debug("AND. key2 step skip list = [%s] ", self.get_step_skip_list(len(self.inverted_index[key2])))
                    else:
                        p2 += 1
                else:
                    # поиск по skip list
                    if self.hasSkip(key1, p1) and (self.getSkip(key1, p1) < posting2[p2]):
                        while self.hasSkip(key1, p1) and (self.getSkip(key1, p1) < posting2[p2]):
                            p1 += self.get_step_skip_list(len(self.inverted_index[key1]))
                            logger.debug("AND. key1 step skip list = [%s] ", self.get_step_skip_list(len(self.inverted_index[key1])))
                    else:
                        p1 += 1
        return result

    

    def postfix(self, infix_tokens):
        """ разбиваем запрос на составляющие"""
        #precendence initialization
        precedence = {}
        precedence['NOT'] = 3
        precedence['AND'] = 2
        precedence['OR'] = 1
        precedence['('] = 0
        precedence[')'] = 0

        output = []
        operator_stack = []

        #creating postfix expression
        for token in infix_tokens:
            if (token == '('):
                operator_stack.append(token)

            elif (token == ')'):
                operator = operator_stack.pop()
                while operator != '(':
                    output.append(operator)
                    operator = operator_stack.pop()

            elif (token in precedence):
                if (operator_stack):
                    current_operator = operator_stack[-1]
                    while (operator_stack and precedence[current_operator] > precedence[token]):
                        output.append(operator_stack.pop())
                        if (operator_stack):
                            current_operator = operator_stack[-1]

                operator_stack.append(token)

            else:
                output.append(token.lower())

        #while staack is not empty appending
        while (operator_stack):
            output.append(operator_stack.pop())
        return output

    def AND_encode(self, posting1, posting2):
        """ AND оператор для pfordelta """
        # [0] список, [1] - token
        key1 = posting1[1]
        key2 = posting2[1]
        posting1 = posting1[0]
        posting2 = posting2[0]
        p1 = 0
        p2 = 0
        result = list()
        if (posting1 is not None) and (posting2 is not None):
            while p1 < self.pl_len(posting1) and p2 < self.pl_len(posting2):
                # поиск по skip list
                if self.pl_get(posting1, p1) == self.pl_get(posting2,p2):
                    result.append(self.pl_get(posting1, p1))
                    p1 += 1
                    p2 += 1
                elif self.pl_get(posting1, p1) > self.pl_get(posting2, p2):
                    # поиск по skip list
                    if self.hasSkip_encode(key2, p2) and (self.getSkip_encode(key2, p2) < self.pl_get(posting1, p1)):
                        while self.hasSkip_encode(key2, p2) and (self.getSkip_encode(key2, p2) < self.pl_get(posting1, p1)):
                            p2 += self.get_step_skip_list(self.pl_len(self.posting_list_encoded[key2]))
                            logger.debug("AND_encode. key2 step skip list = [%s] ", self.get_step_skip_list(self.pl_len(self.posting_list_encoded[key2])))
                    else:
                        p2 += 1
                else:
                    if self.hasSkip_encode(key1, p1) and (self.getSkip_encode(key1, p1) < self.pl_get(posting2, p2)):
                        while self.hasSkip_encode(key1, p1) and (self.getSkip_encode(key1, p1) < self.pl_get(posting2, p2)):
                            p1 += self.get_step_skip_list(self.pl_len(self.posting_list_encoded[key1]))
                            logger.debug("AND_encode. key1 step skip list = [%s] ", self.get_step_skip_list(self.pl_len(self.posting_list_encoded[key1])))
                    else:
                        p1 += 1
        #print(f'AND_encode result = {result}')
        return result

    def OR(posting1, posting2):
        """ OR оператор для list """
        posting1 = posting1[0]
        posting2 = posting2[0]
        p1 = 0
        p2 = 0
        result = list()
        if posting1 is None:
            posting1 = list()
        if posting2 is None:
            posting2 = list()
        while p1 < len(posting1) and p2 < len(posting2):
            if posting1[p1] == posting2[p2]:
                result.append(posting1[p1])
                p1 += 1
                p2 += 1
            elif posting1[p1] > posting2[p2]:
                result.append(posting2[p2])
                p2 += 1
            else:
                result.append(posting1[p1])
                p1 += 1
        while p1 < len(posting1):
            result.append(posting1[p1])
            p1 += 1
        while p2 < len(posting2):
            result.append(posting2[p2])
            p2 += 1
        return result

    def OR_encode(self, posting1, posting2):
        """ OR оператор для pfordelta """
        posting1 = posting1[0]
        posting2 = posting2[0]
        p1 = 0
        p2 = 0
        result = list()

        while p1 < self.pl_len(posting1) and p2 < self.pl_len(posting2):
            if self.pl_get(posting1, p1) == self.pl_get(posting2, p2):
                result.append(self.pl_get(posting1, p1))
                p1 += 1
                p2 += 1
            elif self.pl_get(posting1, p1) > self.pl_get(posting2, p2):
                result.append(self.pl_get(posting2, p2))
                p2 += 1
            else:
                result.append(self.pl_get(posting1, p1))
                p1 += 1
        while p1 < self.pl_len(posting1):
            result.append(self.pl_get(posting1, p1))
            p1 += 1
        while p2 < self.pl_len(posting2):
            result.append(self.pl_get(posting2, p2))
            p2 += 1
        return result

    def NOT(self, posting):
        """ NOT оператор для list """
        posting = posting[0]
        result = list()
        if posting is None:
            posting = list()
        i = 0
        #print(f'posting = {posting}')
        for item in posting:
            while i < item:
                result.append(i)
                i += 1
            else:
                i += 1
        else:
            while i < self.num_docs: #NUM_OF_DOCS:
                result.append(i)
                i += 1
        logger.debug("NOT. self.num_docs = %s", self.num_docs)
        return result

    def NOT_encode(self, posting):
        """ NOT оператор для pfordelta """
        posting = posting[0]
        result = list()
        i = 0
        for item in pfordelta_decode(BitStream(posting)):
            while i < item:
                result.append(i)
                i += 1
            else:
                i += 1
        else:
            while i < self.num_docs: #NUM_OF_DOCS:
                result.append(i)
                i += 1
        logger.debug("NOT_encode. self.num_docs = %s", self.num_docs)
        return result

    #Boolean query processing
    def process_query(self, q):
        """ запуск запроса """
        q = q.replace('(', '( ')
        q = q.replace(')', ' )')
        q = q.split(' ')
        query = []

        for i in q:
            query.append(i) #(ps.stem(i))
        for i in range(0,len(query)):
            if ( query[i]== 'and' or query[i]== 'or' or query[i]== 'not'):
                query[i] = query[i].upper()
        results_stack = []
        postfix_queue = self.postfix(query)

        #return postfix_queue
        #evaluating postfix query expression
        for i in postfix_queue:
            if ( i!= 'AND' and i!= 'OR' and i!= 'NOT'):
                i = i.replace('(', ' ')
                i = i.replace(')', ' ')
                key = i.lower()
                posting_list = self.inverted_index.get(key) #dictionary_inverted.get(i)
                if posting_list is None:
                    key = ''
                #print(f'posting_list = {posting_list}')
                results_stack.append([posting_list, key])
            elif (i=='AND'):
                a = results_stack.pop()
                b = results_stack.pop()
                results_stack.append([self.AND(a, b), '']) #(AND_op(a,b))
            elif (i=='OR'):
                a = results_stack.pop()
                b = results_stack.pop()
                results_stack.append([self.OR(a, b), '']) #(OR_op(a,b))
            elif (i == 'NOT'):
                a = results_stack.pop()
                results_stack.append([self.NOT(a), '']) #(NOT_op(a,doc_ids))

        res =  results_stack.pop()
        return res[0]

    def process_query_encode(self, q):
        """Boolean query processing with encoded list"""
        q = q.replace('(', '( ')
        q = q.replace(')', ' )')
        q = q.split(' ')
        query = []

        for i in q:
            query.append(i) #(ps.stem(i))
        for i in range(0,len(query)):
            if ( query[i]== 'and' or query[i]== 'or' or query[i]== 'not'):
                query[i] = query[i].upper()
        results_stack = []
        postfix_queue = self.postfix(query)

        #return postfix_queue
        #evaluating postfix query expression
        for i in postfix_queue:
            if ( i!= 'AND' and i!= 'OR' and i!= 'NOT'):
                i = i.replace('(', ' ')
                i = i.replace(')', ' ')
                key = i.lower()
                posting_list = self.posting_list_encoded.get(key)
                if posting_list is None:
                    key = ''
                #print(f'encode i = {i}')
                results_stack.append([posting_list, key])
            elif (i=='AND'):
                a = results_stack.pop()
                b = results_stack.pop()
                results_stack.append([self.AND_encode(a, b), ''])
            elif (i=='OR'):
                a = results_stack.pop()
                b = results_stack.pop()
                results_stack.append([self.OR_encode(a, b), ''])
            elif (i == 'NOT'):
                a = results_stack.pop()
                #print(f'query_encode NOT = {pfordelta_decode(BitStream(a))}')
                results_stack.append([self.NOT_encode(a), ''])

        res =  results_stack.pop()
        res = res[0]
        if isinstance(res, bytes):
            res = pfordelta_decode(BitStream(res))
        return res # results_stack.pop()


    def __repr__(self):
        repr_ = f"{self.inverted_index}"
        return repr_

    def __eq__(self, rhs):
        outcome = (
            self.inverted_index == rhs.inverted_index
        )
        return outcome
    
    def load_documents(filepath: str) -> list[str]:
        """Loads documents from file"""
        logger.info("reading documents file...")
        documents = []
        with open(filepath, encoding='utf-8') as file_io:
            i = 0
            for line in file_io.readlines():
                #if i > 1000:
                #    break
                documents.append(line)
                i += 1
        logger.info("reading complete. Read %s documents.", len(documents))
        return documents
    
    # Создание инвертированного индекса
    def build_inverted_index(documents) -> InvertedIndex:
        """ Build inverted index """
        inv_index = InvertedIndex(documents)
        return inv_index
