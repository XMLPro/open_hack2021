from enum import Enum
from sound_keyboard.keyboard.utils.jp_char_util import (
    add_jp_char
)

class State(Enum):
    JAPANESE = 1
    US = 2
    NUMERIC = 3


class Direction(Enum):
    # NAME = (x, y)
    CENTER = (0, 0)
    LEFT = (-1, 0)
    UP = (0, -1)
    RIGHT = (1, 0)
    DOWN = (0, 1)

JP_KEY_MAP = [
    ['あ', 'か', 'さ', 'た', 'な', 'は', 'ま', 'や', 'ら', '小', 'わ', '、']
]

JP_CHILDREN_KEY_MAP = {
    'あ': ['あ', 'い', 'う', 'え', 'お'],
    'か': ['か', 'き', 'く', 'け', 'こ'],
    'さ': ['さ', 'し', 'す', 'せ', 'そ'],
    'た': ['た', 'ち', 'つ', 'て', 'と'],
    'な': ['な', 'に', 'ぬ', 'ね', 'の'],
    'は': ['は', 'ひ', 'ふ', 'へ', 'ほ'],
    'ま': ['ま', 'み', 'む', 'め', 'も'],
    'や': ['や', '(' , 'ゆ', ')' , 'よ'],
    'ら': ['ら', 'り', 'る', 'れ', 'ろ'],
    '小': ['小', '濁', '小', '半', ''  ],
    'わ': ['わ', 'を', 'ん', 'ー', '〜'],
    '、': ['、', '。', '？', '！', '…'],
}

KEYMAP = {
    State.JAPANESE: {
        'parent': JP_KEY_MAP,
        'children': JP_CHILDREN_KEY_MAP,
        'add_char': add_jp_char
    }
}


class KeyboardStateController:

    def __init__(self):
        self.kind = State.JAPANESE
        self.current_parent_char = KEYMAP[self.kind]['parent'][0][0]
        self.current_child_char = KEYMAP[self.kind]['parent'][0][0]
        self.current_parent_position = (0, 0) # (x, y)
        self.current_child_index = 0

        self.selected_parent = False
        self.text = ""
    
    def get_neighbor(self, direction: Direction):
        x, y = self.current_parent_position
        dx, dy = direction.value

        nx = min(max(0, (x + dx)), len(KEYMAP[self.kind]['parent'][0]) - 1)
        ny = min(max(0, (y + dy)), len(KEYMAP[self.kind]['parent']) - 1)

        return (
            KEYMAP[self.kind]['parent'][ny][nx], # neighbor char
            (nx, ny)
        )
    
    def clear(self):
        self.text = ""
        self.move_parent(Direction.CENTER)
        self.selected_parent = False

    def back(self):
        if self.selected_parent:
            self.selected_parent = False
        else:
            self.text = self.text[:-1]
    
    def select(self):
        if self.selected_parent:
            self.text = KEYMAP[self.kind]['add_char'](self.text, self.current_child_char)
            self.move_parent(Direction.CENTER)

        self.selected_parent = not self.selected_parent

    
    def move(self, direction: Direction):
        if not self.selected_parent:
            self.move_parent(direction)
        else:
            self.move_child(direction)


    def move_parent(self, direction: Direction):
        # move current parent char to direction
        char, (nx, ny) = self.get_neighbor(direction)
        
        self.current_parent_char = char
        self.current_parent_position = (nx, ny)
        self.current_child_index = 0
        self.current_child_char = KEYMAP[self.kind]['children'][self.current_parent_char][self.current_child_index]
    

    def move_child(self, direction: Direction):

        index = self.current_child_index
        dx, _ = direction.value

        self.current_child_index = max(0, min(len(KEYMAP[self.kind]['children'][self.current_parent_char]) - 1, index + dx))
        
        self.current_child_char = self.get_child_char(self.current_parent_char, self.current_child_index)
    
    def get_child_char(self, parent_char, index):

        return KEYMAP[self.kind]['children'][parent_char][index]