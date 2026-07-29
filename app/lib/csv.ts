import type { CellValue, DataTable } from "./analysis";

export function parseCsv(text: string, name = "dataset.csv"): DataTable {
  const matrix = parseRows(text.replace(/^\uFEFF/, ""));
  if (matrix.length < 2) throw new Error("The CSV needs a header row and at least one data row.");
  const headers = matrix[0].map((value, index) => value.trim() || `column_${index + 1}`);
  const duplicates = headers.filter((header, index) => headers.indexOf(header) !== index);
  if (duplicates.length) throw new Error(`Duplicate column name: ${duplicates[0]}`);
  const rows = matrix
    .slice(1)
    .filter((values) => values.some((value) => value.trim() !== ""))
    .slice(0, 5000)
    .map((values) =>
      Object.fromEntries(
        headers.map((header, index) => [header, coerce(values[index] ?? "")]),
      ),
    ) as Record<string, CellValue>[];
  if (!rows.length) throw new Error("No data rows were found in the CSV.");
  return { name, headers, rows };
}

export function tableToCsv(table: DataTable): string {
  const lines = [table.headers.map(escapeCell).join(",")];
  for (const row of table.rows) {
    lines.push(table.headers.map((header) => escapeCell(row[header])).join(","));
  }
  return `${lines.join("\n")}\n`;
}

export function downloadText(content: string, filename: string, mime = "text/plain;charset=utf-8") {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function parseRows(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (quoted) {
      if (character === '"' && text[index + 1] === '"') {
        field += '"';
        index += 1;
      } else if (character === '"') {
        quoted = false;
      } else {
        field += character;
      }
      continue;
    }
    if (character === '"' && field.length === 0) {
      quoted = true;
    } else if (character === ",") {
      row.push(field);
      field = "";
    } else if (character === "\n") {
      row.push(field.replace(/\r$/, ""));
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += character;
    }
  }
  if (quoted) throw new Error("The CSV contains an unterminated quoted field.");
  if (field.length || row.length) {
    row.push(field.replace(/\r$/, ""));
    rows.push(row);
  }
  return rows;
}

function coerce(value: string): CellValue {
  const trimmed = value.trim();
  if (!trimmed || /^(na|n\/a|null|none)$/i.test(trimmed)) return null;
  const normalized = trimmed.replace(/,/g, "");
  if (/^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(normalized)) {
    const numeric = Number(normalized);
    if (Number.isFinite(numeric)) return numeric;
  }
  return trimmed;
}

function escapeCell(value: CellValue | undefined): string {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}
