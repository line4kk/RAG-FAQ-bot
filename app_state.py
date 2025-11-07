import json, csv

# Глобальный контейнер состояния
class AppState:
    def __init__(self):
        self.admins = self._load_admins()  # set админов
        self.faq = self._load_faq()  # faq-dict

    def _load_admins(self):
        """Загрузка ID админов из файла"""
        try:
            with open("data\\admin_list.json", "r", encoding="utf-8") as f:
                try:
                    return set(json.load(f))
                except json.decoder.JSONDecodeError as e:
                    return set()
                    
        except FileNotFoundError:
            print("⚠️ Файл admin_list.json не найден. Список админов пуст.")
            return set()

    def _load_faq(self):
        """Загрузка базы данных FAQ. Формат файла - .tsv - текстовый файл, в котором вопросы и ответы на них разделены символом табуляции"""
        try:
            with open("data\\faq.tsv", "r", encoding="utf-8") as f:
                data = dict(csv.reader(f, delimiter="\t"))
        except FileNotFoundError:
            print("⚠️ Файл faq.json не найден. База FAQ пуста.")
            data = {}

        return data

    def reload_admins(self):
        """Обновить список админов"""
        self.admins = self._load_admins()

    def reload_faq(self):
        """Обновить FAQ"""
        self.rag = self._load_faq()


app_state = AppState()

