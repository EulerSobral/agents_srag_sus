import os
import json
import base64
import logging


class UnifyRepository:
    def __init__(
        self,
        source_dir_text: str,
        source_dir_json: str,
        source_dir_png_last_30_days: str,
        source_dir_png_last_12_months: str,
        source_dir_text_delete: str,
        source_dir_json_delete: str,
        output_file: str,
    ):
        try:
            self.source_dir_text = source_dir_text
            self.source_dir_json = source_dir_json
            self.source_dir_png_last_30_days = source_dir_png_last_30_days
            self.source_dir_png_last_12_months = source_dir_png_last_12_months
            self.source_dir_text_delete = source_dir_text_delete
            self.source_dir_json_delete = source_dir_json_delete
            self.output_file = output_file

            content_output_md = self._load_data_file(self.source_dir_text, is_json=False)
            content_output_json = self._load_data_file(self.source_dir_json, is_json=True)
            content_output_png_last_30_days = self._load_png_file(self.source_dir_png_last_30_days)
            content_output_png_last_12_months = self._load_png_file(self.source_dir_png_last_12_months)
            self._delete_file(self.source_dir_text_delete)
            self._delete_file(self.source_dir_json_delete)
            unified = (
                content_output_md
                + "\n\n"
                + "\n## Dados das métricas\n\n"
                + content_output_json
                + "\n\n"
                + "\n## Gráficos da evolução nos últimos 30 dias\n\n"
                + content_output_png_last_30_days
                + "\n\n"
                + "\n## Gráficos da evolução nos últimos 12 meses\n\n"
                + content_output_png_last_12_months
            )
            self._save_data_in_file(unified, self.output_file)
            logging.info("Initialized UnifyRepository successfully.")

        except Exception as e:
            logging.error(f"Error initializing UnifyRepository: {e}")
            raise e

    def _load_data_file(self, file_path: str, is_json: bool) -> str:
        """
        Get the content of a data file, either JSON or plain text.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                if is_json:
                    data = json.load(f)
                    logging.info(f"Loaded JSON data from {file_path}")

                    items = list(data.items())

                    # Collect numeric values to normalize bars (ignore non-numeric)
                    numeric_values = []
                    for _, v in items:
                        try:
                            numeric_values.append(float(v))
                        except Exception:
                            continue

                    max_val = max(numeric_values) if numeric_values else 1.0

                    lines = []
                    lines.append("| Métrica | Valor | Visual |")
                    lines.append("|---|---:|---|")

                    for key, value in items:
                        formatted = str(value)
                        visual = "-"
                        try:
                            val = float(value)
                            # Format with thousands separator and two decimals
                            if "propor" in key.lower() and val <= 100:
                                # likely a percentage - show with 2 decimals and % sign
                                formatted = f"{val:.2f}%"
                            else:
                                formatted = f"{val:,.2f}"

                            # Build a simple horizontal bar (max width 30)
                            width = 30
                            bar_len = int((val / max_val) * width) if max_val > 0 else 0
                            bar_len = max(0, min(width, bar_len))
                            bar = "█" * bar_len + "░" * (width - bar_len)
                            visual = bar
                        except Exception:
                            formatted = str(value)
                            visual = "-"

                        # Escape pipe characters in key/value to avoid breaking the table
                        safe_key = str(key).replace("|", "\\|")
                        safe_value = str(formatted).replace("|", "\\|")
                        lines.append(f"| {safe_key} | {safe_value} | {visual} |")

                    return "\n".join(lines)
                else:
                    content = f.read()
                    logging.info(f"Loaded text data from {file_path}")
                    return content
        except Exception as e:
            logging.error(f"Error loading data from {file_path}: {e}")
            raise e

    def _save_data_in_file(self, content: str, output_file: str):
        """
        Save the unified content into the output markdown file.
        """
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(content)
            logging.info(f"Unified content saved to {output_file}")
        except Exception as e:
            logging.error(f"Error saving unified content to {output_file}: {e}")
            raise e

    def _delete_file(self, file_path: str):
        """
        Delete a file if it exists.
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"Deleted file: {file_path}")
        except Exception as e:
            logging.error(f"Error deleting file {file_path}: {e}")
            raise e

    def _load_png_file(self, png_path: str) -> str:
        """
        Load a PNG file and return it as a Markdown image link.
        """
        try:
            rel_path = os.path.basename(png_path)
            return f"![Imagem: {rel_path}]({rel_path})\n\n"
        except Exception as e:
            logging.error(f"Error referencing PNG from {png_path}: {e}")
            return f"Erro ao referenciar imagem: {e}\n\n"