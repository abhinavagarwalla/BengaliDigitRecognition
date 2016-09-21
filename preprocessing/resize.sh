filename="$1"
size="$2"
rm images_resized_"$size"
mkdir images_resized_"$size"
while read -r line
do
	convert ../data/images/"$line" -colorspace RGB +sigmoidal-contrast 11.6933 -define filter:filter=Sinc -define filter:window=Jinc -define filter:lobes=3 -resize "$size"x"$size"! -sigmoidal-contrast 11.6933 -colorspace sRGB ../data/images_resized_"$size"/"$line"
done < "$filename"
