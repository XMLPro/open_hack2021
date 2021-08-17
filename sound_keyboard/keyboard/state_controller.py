from enum import Enum

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
    ['あ', 'か', 'さ'],
    ['た', 'な', 'は'],
    ['ま', 'や', 'ら'],
    ['小', 'わ', '、'],
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
    '小': ['小', '濁', '小', '半', ' ' ],
    'わ': ['わ', 'を', 'ん', 'ー', '〜'],
    '、': ['、', '。', '？', '！', '…'],
}

KEYMAP = {
    State.JAPANESE: {
        'parent': JP_KEY_MAP,
        'children': JP_CHILDREN_KEY_MAP
    }
}


class KeyboardStateController:

    def __init__(self):
        self.kind = State.JAPANESE
        self.current_parent_char = KEYMAP[self.kind]['parent'][0][0]
        self.current_child_char = KEYMAP[self.kind]['parent'][0][0]
        self.current_parent_position = (0, 0) # (x, y)
        self.current_child_position = Direction.CENTER
    
    def get_neighbor(self, direction: Direction):
        x, y = self.current_parent_position
        dx, dy = direction.value

        nx = (x + dx) % len(KEYMAP[self.kind]['parent'][0])
        ny = (y + dy) % len(KEYMAP[self.kind]['parent'])

        return (
            KEYMAP[self.kind]['parent'][ny][nx], # neighbor char
            (nx, ny)
        )

    def move(self, direction: Direction):
        # move current parent char to direction
        char, (nx, ny) = self.get_neighbor(direction)
        
        self.current_parent_char = self.current_child_char = char
        self.current_parent_position = (nx, ny)
        self.current_child_position = Direction.CENTER
    

    def move_child(self, direction: Direction):

        x, y = self.current_child_position
        dx, dy = direction.value

        if (x + dx, y + dy) != (0, 0):
            self.current_child_position = direction
        else:
            self.current_child_position = Direction.CENTER
        
        self.current_child_char = self.get_child_char(self.current_parent_char, self.current_child_position)
    
    def get_child_char(self, parent_char, direction: Direction):

        index = 0
        if direction == Direction.CENTER:
            index = 0
        if direction == Direction.LEFT:
            index = 1
        if direction == Direction.UP:
            index = 2
        if direction == Direction.RIGHT:
            index = 3
        if direction == Direction.DOWN:
            index = 4
        
        return KEYMAP[self.kind]['children'][parent_char][index]