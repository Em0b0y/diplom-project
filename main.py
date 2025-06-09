import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import tkinter as tk
from sklearn.metrics import classification_report, confusion_matrix
from tkinter import ttk, scrolledtext, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns


class MLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Машинне навчання. Детект. Слижук Максим")
        self.root.geometry("900x750")


        self.dataset_path = tk.StringVar()


        self.ml_methods = ["Isolation Forest", "KNN", "K-Means"]


        self.vulnerability_types = ["Reconnaissance", "Fuzzers", "Generic", "Backdoor", "DoS", "Analysis", "Exploits"]
        self.vulnerability_features = {
            "Reconnaissance": [
                'sttl', 'ct_srv_dst', 'ct_srv_src', 'rate', 'dmean',
                'ct_dst_src_ltm', 'sloss', 'is_sm_ips_ports', 'sinpkt', 'smean'
            ],
            "Backdoor2": [
                'sttl', 'ct_srv_dst', 'ct_srv_src', 'rate', 'dmean',
                'ct_dst_src_ltm', 'sloss', 'is_sm_ips_ports', 'sinpkt', 'smean'
            ],
            "Exploits": [
                'sttl', 'ct_srv_dst', 'ct_srv_src', 'rate', 'dmean',
                'ct_dst_src_ltm', 'sloss', 'is_sm_ips_ports', 'sinpkt', 'smean'
            ],
            "Fuzzers": [
                'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'ct_src_dport_ltm', 'ct_srv_dst',
                'ct_srv_src', 'sttl', 'ct_dst_ltm', 'ct_src_ltm', 'swin', 'ct_state_ttl'
            ],
            "Generic": [
                'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'ct_src_dport_ltm', 'ct_srv_dst',
                'ct_srv_src', 'sttl', 'ct_dst_ltm', 'ct_src_ltm', 'swin', 'ct_state_ttl'
            ],
            "Backdoor": [
                'sttl', 'rate', 'ct_state_ttl', 'swin', 'dwin',
                'dtcpb', 'stcpb', 'dmean', 'sload', 'sloss'
            ],
            "DoS": [
                'sttl', 'ct_state_ttl', 'rate', 'swin', 'dwin',
                'stcpb', 'dtcpb', 'dmean', 'sload', 'dload'
            ],
            "Analysis": [
                'rate', 'ct_flw_http_mthd', 'sttl', 'ct_state_ttl', 'swin',
                'dwin', 'dmean', 'trans_depth', 'stcpb', 'dtcpb'
            ]
        }

        self.create_widgets()

    def create_widgets(self):

        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.grid(row=0, column=0, sticky="ew")
        self.root.columnconfigure(0, weight=1)  # Allow control frame to expand

        ttk.Button(control_frame, text="Вибрати датасет", command=self.load_dataset).grid(row=0, column=0, padx=5,
                                                                                          pady=5, sticky="w")
        self.dataset_label = ttk.Label(control_frame, text="Файл не вибрано")
        self.dataset_label.grid(row=0, column=1, columnspan=2, padx=5, pady=5, sticky="w")


        ttk.Label(control_frame, text="Виберіть метод ML:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.ml_method_var = tk.StringVar()
        ml_method_menu = ttk.Combobox(control_frame, textvariable=self.ml_method_var, values=self.ml_methods, width=25)
        ml_method_menu.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        ml_method_menu.current(0)


        ttk.Label(control_frame, text="Виберіть тип вразливостей:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.vulnerability_var = tk.StringVar()
        vulnerability_menu = ttk.Combobox(control_frame, textvariable=self.vulnerability_var,
                                          values=self.vulnerability_types, width=25)
        vulnerability_menu.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        vulnerability_menu.current(0)


        ttk.Button(control_frame, text="Запустити аналіз", command=self.run_analysis).grid(row=3, column=0,
                                                                                           columnspan=2, pady=10)

        control_frame.columnconfigure(1, weight=1)


        self.result_text = scrolledtext.ScrolledText(self.root, width=100, height=15, wrap=tk.WORD)
        self.result_text.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")


        self.figure = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax = None
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")


        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=1)

    def load_dataset(self):
        file_path = filedialog.askopenfilename(
            title="Виберіть CSV файл датасету",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            self.dataset_path.set(file_path)
            self.dataset_label.config(text=file_path.split('/')[-1])
        else:
            self.dataset_label.config(text="Файл не вибрано")
            messagebox.showwarning("Попередження", "Файл не вибрано!")

    def run_analysis(self):
        self.result_text.delete(1.0, tk.END)

        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Confusion Matrix")
        self.canvas.draw()

        # Get user selections
        dataset_path = self.dataset_path.get()
        if not dataset_path:
            messagebox.showerror("Помилка", "Спочатку виберіть датасет!")
            return

        ml_method = self.ml_method_var.get()
        vulnerability_type = self.vulnerability_var.get()

        self.result_text.insert(tk.END,
                                f"Запуск аналізу...\nМетод: {ml_method}\nТип вразливості: {vulnerability_type}\nДатасет: {dataset_path.split('/')[-1]}\n\n")
        self.root.update_idletasks()

        try:

            df = pd.read_csv(dataset_path, low_memory=False)
            df.columns = df.columns.str.strip()


            if vulnerability_type not in self.vulnerability_features:
                messagebox.showerror("Помилка", f"Невідомий тип вразливості: {vulnerability_type}")
                return
            features = self.vulnerability_features[vulnerability_type]


            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                messagebox.showerror("Помилка", f"Наступні ознаки відсутні в датасеті: {', '.join(missing_features)}")
                return


            X_raw = df[features].copy()


            for col in X_raw.columns:
                if X_raw[col].dtype == 'object':

                    X_raw[col] = pd.to_numeric(X_raw[col], errors='coerce')
            X_raw = X_raw.replace([np.inf, -np.inf], np.nan)


            original_indices_of_X_raw = X_raw.index
            X_raw.dropna(inplace=True)


            valid_indices_after_dropna = X_raw.index

            if 'label' not in df.columns and 'attack_cat' not in df.columns:
                messagebox.showerror("Помилка", "Датасет повинен містити колонку 'label' або 'attack_cat'.")
                return

            if vulnerability_type == "Reconnaissance" and 'label' in df.columns:
                y_series = df.loc[valid_indices_after_dropna, 'label']
            elif 'attack_cat' in df.columns:
                y_series = df.loc[valid_indices_after_dropna, 'attack_cat'].apply(
                    lambda x: 1 if x == vulnerability_type else 0)
            else:
                y_series = df.loc[valid_indices_after_dropna, 'label']


            normal_indices_in_y = y_series[y_series == 0].index
            attack_indices_in_y = y_series[y_series == 1].index

            self.result_text.insert(tk.END,
                                    f"Розподіл класів перед балансуванням:\n{y_series.value_counts(normalize=True)}\n\n")

            if len(attack_indices_in_y) == 0:
                messagebox.showwarning("Попередження",
                                       f"Не знайдено екземплярів для типу вразливості '{vulnerability_type}'. Аналіз може бути неінформативним.")


            np.random.seed(42)

            num_attacks = len(attack_indices_in_y)
            num_normal_to_keep = min(len(normal_indices_in_y),
                                     num_attacks * 2)

            if len(normal_indices_in_y) > 0 and num_normal_to_keep > 0:
                normal_to_keep_indices = np.random.choice(normal_indices_in_y, num_normal_to_keep, replace=False)
                selected_final_indices = np.concatenate([normal_to_keep_indices, attack_indices_in_y])
            elif len(attack_indices_in_y) > 0:
                selected_final_indices = attack_indices_in_y
            else:
                messagebox.showerror("Помилка", "Немає даних для аналізу після фільтрації та вибору міток.")
                return

            X = X_raw.loc[selected_final_indices]
            y = y_series.loc[selected_final_indices]

            self.result_text.insert(tk.END,
                                    f"Розподіл класів після балансування:\n{y.value_counts(normalize=True)}\n\n")
            self.root.update_idletasks()

            if X.empty or len(X) < 2:
                messagebox.showerror("Помилка",
                                     "Недостатньо даних для аналізу після попередньої обробки та балансування.")
                return


            transformer = PowerTransformer(method='yeo-johnson', standardize=False)
            X_transformed = transformer.fit_transform(X)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_transformed)


            X_indices_for_split = np.arange(len(X_scaled))

            if len(np.unique(y)) > 1:
                stratify_option = y
            else:
                stratify_option = None

            train_indices_in_X, test_indices_in_X, y_train, y_test = train_test_split(
                X_indices_for_split, y, test_size=0.3, random_state=42, stratify=stratify_option
            )
            X_train = X_scaled[train_indices_in_X]
            X_test = X_scaled[test_indices_in_X]


            original_df_indices_for_test_set = X.index[test_indices_in_X]

            y_pred = None


            if ml_method == "Isolation Forest":
                contamination_value = 'auto'
                if y.value_counts(normalize=True).get(1, 0) > 0.4:
                    contamination_value = min(0.49, y.value_counts(normalize=True).get(1, 0))
                self.result_text.insert(tk.END, f"Isolation Forest contamination: {contamination_value}\n")
                model = IsolationForest(contamination=contamination_value, random_state=42)
                model.fit(X_train)
                labels = model.predict(X_test)
                y_pred = np.where(labels == -1, 1, 0)
                report_title = f"=== {ml_method} Детект Аномалій ==="

            elif ml_method == "KNN":
                model = KNeighborsClassifier(n_neighbors=5)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                report_title = f"=== {ml_method} Класифікація ==="

            elif ml_method == "K-Means":
                model = KMeans(n_clusters=2, random_state=42, n_init='auto')
                labels = model.fit_predict(X_test)
                if len(y_test) > 0 and len(labels) == len(y_test):
                    cm_kmeans = confusion_matrix(y_test, labels)
                    cluster0_is_vuln = y_test[labels == 0].mean() > 0.5
                    cluster1_is_vuln = y_test[labels == 1].mean() > 0.5
                    if pd.Series(labels).value_counts().get(0, 0) > pd.Series(labels).value_counts().get(1,0):
                        if not cluster0_is_vuln and cluster1_is_vuln:
                            y_pred = labels
                        else:
                            y_pred = 1 - labels
                    else:
                        if not cluster1_is_vuln and cluster0_is_vuln:
                            y_pred = 1 - labels
                        else:
                            y_pred = labels
                else:
                    y_pred = labels

                report_title = f"=== {ml_method} Кластеризація (Позначена) ==="

            if y_pred is not None and len(y_test) > 0:
                try:
                    report = classification_report(y_test, y_pred, zero_division=0)
                    self.result_text.insert(tk.END, f"{report_title}\n{report}\n")

                    vulnerable_indices_in_y_pred = np.where(y_pred == 1)[0]

                    if len(vulnerable_indices_in_y_pred) > 0:
                        actual_vulnerable_df_indices = original_df_indices_for_test_set[vulnerable_indices_in_y_pred]


                        original_vulnerable_logs = df.loc[actual_vulnerable_df_indices]

                        output_file = f"potential_vulnerable_logs_{vulnerability_type.replace(' ', '_')}_{ml_method.replace(' ', '_')}.csv"
                        original_vulnerable_logs.to_csv(output_file, index=False)
                        self.result_text.insert(tk.END,
                                                f"\nПотенційно вразливі логи ({len(original_vulnerable_logs)}) збережено у: {output_file}\n")
                    else:
                        self.result_text.insert(tk.END, "\nНе знайдено потенційно вразливих логів для збереження.\n")

                except ValueError as e_report:
                    self.result_text.insert(tk.END,
                                            f"Помилка генерації звіту: {e_report}\nМожливо, в тестових даних лише один клас після обробки.\n")
            elif len(y_test) == 0:
                self.result_text.insert(tk.END, "Немає тестових даних для генерації звіту.\n")

            if y_pred is not None and len(y_test) > 0 and len(y_pred) == len(y_test):
                self.ax.clear()
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=self.ax,
                            cbar=True)
                self.ax.set_title(f'Confusion Matrix ({ml_method})')
                self.ax.set_xlabel('Predicted')
                self.ax.set_ylabel('True')
                self.canvas.draw()
            elif self.ax:
                self.ax.clear()
                self.ax.text(0.5, 0.5, "Немає даних для Confusion Matrix", ha='center', va='center')
                self.canvas.draw()


        except FileNotFoundError:
            messagebox.showerror("Помилка", f"Файл не знайдено: {dataset_path}")
            self.result_text.insert(tk.END, f"Помилка: Файл не знайдено: {dataset_path}\n")
        except pd.errors.EmptyDataError:
            messagebox.showerror("Помилка", "Обраний файл порожній або не є валідним CSV.")
            self.result_text.insert(tk.END, "Помилка: Обраний файл порожній.\n")
        except KeyError as e:
            messagebox.showerror("Помилка", f"Помилка ключа (можливо, відсутня колонка в CSV): {str(e)}")
            self.result_text.insert(tk.END, f"Помилка ключа: {str(e)}\n")
        except Exception as e:
            messagebox.showerror("Помилка", f"Виникла неочікувана помилка: {str(e)}")
            self.result_text.insert(tk.END, f"Неочікувана помилка: {str(e)}\n")
            import traceback
            self.result_text.insert(tk.END, f"\nTraceback:\n{traceback.format_exc()}\n")
        finally:
            self.result_text.insert(tk.END, "\nАналіз завершено.\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = MLApp(root)
    root.mainloop()
