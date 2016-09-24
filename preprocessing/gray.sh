filename="$1"
size="$2"
rm ../data/images_gray_"$size" -r
mkdir ../data/images_gray_"$size"
while read -r line
do
	convert ../data/images_resized_"$size"/"$line" -type Grayscale -density 300 ../data/images_gray_"$size"/"$line"
done < "$filename"
