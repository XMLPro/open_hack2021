import unicodedata

def is_voicing_diacritic(char):

    return char.encode('utf-8') in [
        b"\xe3\x82\x99",
        b"\xe3\x82\x9b",
        b"\xef\xbe\x9e",
    ]

def is_devoicing_diacritic(char):

    return char.encode('utf-8') in [
        b"\xe3\x82\x9a",
        b"\xe3\x82\x9c",
        b"\xef\xbe\x9f",
    ]

def add_voicing_diacritic(char):
    
    out = char
    if char in [
        'あ', 'い', 'う', 'え', 'お',
        'か', 'き', 'く', 'け', 'こ',
        'さ', 'し', 'す', 'せ', 'そ',
        'た', 'ち', 'つ', 'て', 'と',
        'な', 'に', 'ぬ', 'ね', 'の',
        'は', 'ひ', 'ふ', 'へ', 'ほ',
        'ま', 'み', 'む', 'め', 'も',
        'や', 'ゆ', 'よ',
        'ら', 'り', 'る', 'れ', 'ろ',
        'わ', 'を', 'ん'
    ]:
        out = (out.encode('utf-8') + b"\xe3\x82\x99").decode('utf-8')
    
    return out

def add_devoicing_diacritic(char):
    out = char
    if char in [
        'は', 'ひ', 'ふ', 'へ', 'ほ',
    ]:
        out = (out.encode('utf-8') + b"\xe3\x82\x9a").decode('utf-8')
    
    return out

def add_jp_char(text, char):
    out = text
    if char == '小':
        pass
    elif char == '濁':
        if is_devoicing_diacritic(text[-1]):
            text = text[:-1]

        if is_voicing_diacritic(text[-1]):
            # すでに濁点がついているならはずす
            out = text[:-1]
        else:
            # 濁点をつける
            out = text[:-1] + add_voicing_diacritic(text[-1])
    elif char == '半':
        if is_voicing_diacritic(text[-1]):
            text = text[:-1]
        if is_devoicing_diacritic(text[-1]):
            # すでに濁点がついているならはずす
            out = text[:-1]
        else:
            # 濁点をつける
            out = text[:-1] + add_devoicing_diacritic(text[-1])
    elif char == '小':
        pass

    else:
        out += char
    
    return out