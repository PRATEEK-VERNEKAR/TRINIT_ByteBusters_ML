const mongoose=require("mongoose")

const Schema = mongoose.Schema;

const textSchmema = new Schema({
    name:{
        type:String
    }
    ,
    dataArray: [
        {
            link:String,
            text:String,
            time:String,
            date:String,
            commenting:Boolean,
            staring:Boolean,
            groping:Boolean,
            severity:Number
        }
    ]
});


const Text=new mongoose.model('text',textSchmema)
module.exports=Text;