#!/bin/bash

# remove all img except logo
find img/* | grep -v "icon.png" | xargs rm

# remove ui uncompiled
find ui/* | grep -v ".ui" | xargs rm

#
rm resources.qrc


echo -n "Type bersion number (e.g. 2.5)"
read version
zip -r ../dzetsaka$version.zip .

