filename="$1"
size="$2"
rm ../data/images_otsu_"$size" -r
mkdir ../data/images_otsu_"$size"
while read -r line
do
	#convert ../data/images_resized_"$size"/"$line" -monochrome -density 300 ../data/images_binary_"$size"/"$line"
	#convert ../data/images/"$line" -monochrome -density 300 ../data/images_binary/"$line"
	./otsu ../data/images_resized_"$size"/"$line" ../data/images_otsu_"$size"/"$line"
done < "$filename"
