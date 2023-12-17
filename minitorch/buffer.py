from __future__ import annotations

class MiniBuffer:
    def __init__(self, data: list[list[float]]) -> None:
        assert all(len(row) == len(data[0]) for row in data), "Cannot create MiniBuffer. All rows must have the same length."
        self.data = data

    def add(self, other: MiniBuffer) -> MiniBuffer:
        out_data = []

        for x_row, y_row in zip(self.data, other.data):
            out_row = []

            for x_val, y_val in zip(x_row, y_row):
                out_row.append(x_val + y_val)

            out_data.append(out_row)

        return MiniBuffer(out_data)
    
    def sub(self, other: MiniBuffer) -> MiniBuffer:
        out_data = []

        for x_row, y_row in zip(self.data, other.data):
            out_row = []

            for x_val, y_val in zip(x_row, y_row):
                out_row.append(x_val - y_val)

            out_data.append(out_row)

        return MiniBuffer(out_data)

    def mul(self, other: MiniBuffer) -> MiniBuffer:
        out_data = []

        for x_row in self.data:
            out_row = []

            for x_val in x_row:
                out_row.append(x_val * other.data[0][0])

            out_data.append(out_row)

        return MiniBuffer(out_data)
    
    def div(self, other: MiniBuffer) -> MiniBuffer:
        out_data = []

        for x_row in self.data:
            out_row = []

            for x_val in x_row:
                out_row.append(x_val / other.data[0][0])

            out_data.append(out_row)

        return MiniBuffer(out_data)
    
    def __repr__(self) -> str:
        repr = str("([")

        for row in self.data:
            if not row == self.data[0]:
                repr += "          ["
            else:
                repr += "["
            
            for value in row:
                value_str = f"{value:.4f}"
                
                if value > 0:
                    # Indent to align with '-' character of negative numbers
                    value_str = " " + value_str
                    
                if not value == row[-1]:
                    repr += value_str + ", "
                else:
                    repr += value_str

            if not row == self.data[-1]:
                repr += "],\n"
            else:
                repr += "]"

        repr += "])"

        return repr