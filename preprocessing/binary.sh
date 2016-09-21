filename="$1"
#size="$2"
rm images_binary
mkdir images_binary
while read -r line
do
	#convert ../data/images_resized_"$size"/"$line" -monochrome -density 300 ../data/images_binary_"$size"/"$line"
	convert ../data/images/"$line" -monochrome -density 300 ../data/images_binary/"$line"
done < "$filename"
