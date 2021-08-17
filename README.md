# open_hack2021
OpenHackU 2021 develop repository

### 環境

環境の構築にはpipenvを使います。pipenvは

```
python3 -m pip install pipenv
```

でインストールできます。

Pipfileにしたがって依存をすべてインストールしたい場合は`pipenv install`、新規にパッケージをインストールしたい場合は`pipenv install <パッケージ名>`で追加してください。

開発環境でのみ使いたいパッケージをインストールしたい場合は`pipenv install --dev <パッケージ名>`でインストール可能です。

作られた環境には`pipenv shell`で入れます。

# キーボードの実行

```
pipenv run keyboard
```

Pipfileにscriptsとして登録されています。他にもショートカットを設定したい場合はここを利用してください。