const mongoose=require('mongoose');
const dotenv=require('dotenv');

dotenv.config({path:'./config.env'});

const DB="mongodb+srv://yogeshhtnavi:R7oVZnYREoCmiI57@cluster0.w5aaimx.mongodb.net/NITKHack?retryWrites=true&w=majority";


mongoose.connect(DB,{
    useNewUrlParser:true,
    useUnifiedTopology:true,
}).then(()=>{
    console.log("Connection Successful");
}).catch((err)=>{
    console.log(err);
})