version=`grep -Po '(?<=version=).*' metadata.txt`

cd ../

zip -FSr dzetsaka_${version}.zip dzetsaka/* -x "dzetsaka/__pycache__/**" "dzetsaka/**.qrc" "dzetsaka/**.ui*" "dzetsaka/**__pycache__**" "dzetsaka/**.sh"
