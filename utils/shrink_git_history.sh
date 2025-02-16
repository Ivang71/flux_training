OLD_NAME=$(git config --global user.name)
OLD_EMAIL=$(git config --global user.email)

NEW_NAME="Ivang71"
NEW_EMAIL="artemtolv@gmail.com"

git config --global user.name "$NEW_NAME"
git config --global user.email "$NEW_EMAIL"

git reset $(git commit-tree HEAD^{tree} -m "init")

git config --global user.name "$OLD_NAME"
git config --global user.email "$OLD_EMAIL"